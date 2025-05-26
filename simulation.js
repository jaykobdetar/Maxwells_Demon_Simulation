// Physical constants and unit system
// Using normalized units: distance in pixels, time in seconds
// kT = 1 at T = 300K, mass = 1
const k_B = 1.0; // Normalized Boltzmann constant
const T_REF = 300; // Reference temperature in Kelvin
const MASS = 1.0; // Particle mass in normalized units

// For entropy calculations, we need Planck's constant in our unit system
// Physical constants for argon at 300K:
// h = 6.626e-34 J⋅s, m_Ar = 6.6e-26 kg, k_B = 1.38e-23 J/K
// λ_thermal = h/√(2πmkT) ≈ 1.6e-11 m at 300K
//
// In our normalized units:
// - Length: 1 pixel ≈ 10 nm (chamber width ~800px represents ~8μm)
// - Velocity: 100 px/s = √(k_B⋅T_REF/m) for argon at 300K
// - Energy: k_B⋅T_REF = 1 (normalized)
//
// Converting λ_thermal to pixel units: 1.6e-11 m / 10e-9 m/px = 0.0016 px
// Therefore h²/(2πmk_B) in normalized units:
const PIXEL_TO_METER = 10e-9; // 1 pixel = 10 nm
const LAMBDA_THERMAL_METERS = 1.6e-11; // meters at 300K for argon
const LAMBDA_THERMAL_PIXELS = LAMBDA_THERMAL_METERS / PIXEL_TO_METER;
const H2_OVER_2PI_MK = LAMBDA_THERMAL_PIXELS * LAMBDA_THERMAL_PIXELS; // ≈ 2.56e-6 px²

// Simulation constants
const VELOCITY_SCALE = 100; // pixels per unit velocity
const GATE_WIDTH = 20; // pixels
const PARTICLE_RADIUS = 3; // pixels
const MAX_TIMESTEP = 0.01; // seconds
const COLLISION_CELL_SIZE = 20; // pixels for spatial hashing
const GATE_LOOK_AHEAD_TIME = 0.15; // seconds
const GATE_DECISION_DISTANCE = 40; // pixels
const DEMON_MEMORY_LIMIT = 50; // bits
const MEASUREMENT_NOISE = 0.05; // 5% measurement error for realistic demon

// Visualization constants
const HISTOGRAM_X = 20;
const HISTOGRAM_Y_OFFSET = 170;
const HISTOGRAM_WIDTH = 200;
const HISTOGRAM_HEIGHT = 150;
const HISTOGRAM_BINS = 25;

const PHASE_SPACE_X_OFFSET = 220;
const PHASE_SPACE_Y_OFFSET = 170;
const PHASE_SPACE_WIDTH = 200;
const PHASE_SPACE_HEIGHT = 150;

// Collision effect animation
class CollisionEffect {
    constructor(x, y, energy) {
        this.x = x;
        this.y = y;
        this.radius = 0;
        this.maxRadius = 10 + energy * 5;
        this.opacity = 0.8;
        this.color = energy > 1 ? `rgba(255, 150, 100, ` : `rgba(100, 150, 255, `;
        this.lifetime = 0;
        this.maxLifetime = 0.3; // seconds
    }
    
    update(dt) {
        this.lifetime += dt;
        const progress = this.lifetime / this.maxLifetime;
        
        if (progress >= 1) {
            return false; // Remove effect
        }
        
        // Expand and fade
        this.radius = this.maxRadius * Math.sin(progress * Math.PI);
        this.opacity = 0.8 * (1 - progress);
        
        return true; // Keep effect
    }
    
    render(ctx) {
        ctx.save();
        ctx.globalAlpha = this.opacity;
        ctx.strokeStyle = this.color + this.opacity + ')';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.stroke();
        
        // Inner glow
        const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
        gradient.addColorStop(0, this.color + (this.opacity * 0.3) + ')');
        gradient.addColorStop(1, this.color + '0)');
        ctx.fillStyle = gradient;
        ctx.fill();
        
        ctx.restore();
    }
}

// Demon memory management strategies
class DemonMemory {
    constructor(capacity, strategy = 'fifo') {
        this.capacity = capacity;
        this.strategy = strategy;
        this.storage = [];
        this.accessCount = new Map(); // For LRU
        this.totalAccesses = 0;
        this.erasureCount = 0;
        this.compressionRatio = 4; // For compressed strategy
    }
    
    store(measurement) {
        const timestamp = Date.now();
        const entry = {
            ...measurement,
            timestamp,
            accessCount: 0,
            compressed: false
        };
        
        // Handle different storage strategies
        switch (this.strategy) {
            case 'fifo':
                this.storeFIFO(entry);
                break;
            case 'lru':
                this.storeLRU(entry);
                break;
            case 'selective':
                this.storeSelective(entry);
                break;
            case 'compressed':
                this.storeCompressed(entry);
                break;
        }
        
        return this.getStorageCost();
    }
    
    storeFIFO(entry) {
        this.storage.push(entry);
        if (this.storage.length > this.capacity) {
            this.storage.shift();
            this.erasureCount++;
        }
    }
    
    storeLRU(entry) {
        this.storage.push(entry);
        if (this.storage.length > this.capacity) {
            // Find least recently used
            let lruIndex = 0;
            let minAccess = Infinity;
            for (let i = 0; i < this.storage.length - 1; i++) {
                if (this.storage[i].accessCount < minAccess) {
                    minAccess = this.storage[i].accessCount;
                    lruIndex = i;
                }
            }
            this.storage.splice(lruIndex, 1);
            this.erasureCount++;
        }
    }
    
    storeSelective(entry) {
        // Calculate information value (higher measurement error = lower value)
        const infoValue = 1 / (1 + entry.measurementError);
        entry.infoValue = infoValue;
        
        this.storage.push(entry);
        if (this.storage.length > this.capacity) {
            // Remove entry with lowest information value
            let minValueIndex = 0;
            let minValue = Infinity;
            for (let i = 0; i < this.storage.length; i++) {
                if (this.storage[i].infoValue < minValue) {
                    minValue = this.storage[i].infoValue;
                    minValueIndex = i;
                }
            }
            this.storage.splice(minValueIndex, 1);
            this.erasureCount++;
        }
    }
    
    storeCompressed(entry) {
        // Compress by reducing precision
        if (this.storage.length >= this.capacity * 0.8) {
            entry.compressed = true;
            entry.compressedSpeed = Math.round(entry.measuredSpeed / 10) * 10;
            entry.originalBits = 1;
            entry.compressedBits = 1 / this.compressionRatio;
        }
        
        this.storage.push(entry);
        
        // When full, remove oldest compressed entries first
        if (this.storage.length > this.capacity) {
            const compressedIndex = this.storage.findIndex(e => e.compressed);
            if (compressedIndex !== -1) {
                this.storage.splice(compressedIndex, 1);
            } else {
                this.storage.shift();
            }
            this.erasureCount++;
        }
    }
    
    access(index) {
        if (index >= 0 && index < this.storage.length) {
            this.storage[index].accessCount++;
            this.totalAccesses++;
            return this.storage[index];
        }
        return null;
    }
    
    getStorageCost() {
        // Calculate total information stored
        let totalBits = 0;
        for (const entry of this.storage) {
            if (entry.compressed) {
                totalBits += entry.compressedBits || 0.25;
            } else {
                totalBits += 1;
            }
        }
        return totalBits;
    }
    
    getErasureCost(efficiencyFactor = 1.0) {
        // Each erasure costs ln(2) in normalized units
        // In realistic mode, multiply by efficiency factor (>1) for imperfect erasure
        return this.erasureCount * Math.log(2) * efficiencyFactor;
    }
    
    getUtilization() {
        return this.storage.length / this.capacity;
    }
    
    getStatistics() {
        return {
            strategy: this.strategy,
            stored: this.storage.length,
            capacity: this.capacity,
            utilization: this.getUtilization(),
            erasures: this.erasureCount,
            totalAccesses: this.totalAccesses,
            storageCost: this.getStorageCost(),
            erasureCost: this.getErasureCost()
        };
    }
    
    clear() {
        const clearedCount = this.storage.length;
        this.storage = [];
        this.erasureCount += clearedCount;
        this.totalAccesses = 0;
        return clearedCount * Math.log(2); // Return erasure cost
    }
}

class Particle {
    constructor(x, y, temperature) {
        this.x = x;
        this.y = y;
        this.radius = PARTICLE_RADIUS;
        this.vx = 0;
        this.vy = 0;
        
        // Generate velocity from Maxwell-Boltzmann distribution
        this.generateMaxwellBoltzmannSpeed(temperature);
        
        this.trail = [];
        this.maxTrailLength = 20;
    }
    
    generateMaxwellBoltzmannSpeed(temperature) {
        // For 2D Maxwell-Boltzmann distribution:
        // Each velocity component is Gaussian with σ = sqrt(kT/m)
        // Speed magnitude follows chi distribution with 2 degrees of freedom
        
        // Generate two independent Gaussian components using Box-Muller
        const u1 = Math.random();
        const u2 = Math.random();
        const R = Math.sqrt(-2 * Math.log(u1));
        const theta = 2 * Math.PI * u2;
        
        // Velocity components with proper scaling
        const sigma = Math.sqrt(k_B * temperature / (MASS * T_REF));
        const vx_normalized = R * Math.cos(theta) * sigma;
        const vy_normalized = R * Math.sin(theta) * sigma;
        
        // Convert to pixels/second (visual scaling)
        this.vx = vx_normalized * VELOCITY_SCALE;
        this.vy = vy_normalized * VELOCITY_SCALE;
        
        return this.speed;
    }
    
    get speed() {
        return Math.sqrt(this.vx * this.vx + this.vy * this.vy);
    }
    
    update(dt, slowMotion) {
        const effectiveDt = slowMotion ? dt * 0.1 : dt;
        
        if (slowMotion) {
            this.trail.push({ x: this.x, y: this.y });
            if (this.trail.length > this.maxTrailLength) {
                this.trail.shift();
            }
        } else {
            this.trail = [];
        }
        
        // Velocity Verlet integration for perfect energy conservation
        const actualDt = Math.min(effectiveDt, MAX_TIMESTEP);
        
        // For free particles, acceleration is 0 (no external forces)
        // Position update: x(t+dt) = x(t) + v(t)*dt
        this.x += this.vx * actualDt;
        this.y += this.vy * actualDt;
        
        // Velocity remains constant for free flight
        // Collisions will handle velocity changes
    }
    
    getColor(meanSpeed) {
        const speedRatio = this.speed / meanSpeed;
        const isFast = speedRatio > 1;
        
        if (isFast) {
            // Fast particles: warm colors (red to yellow)
            const intensity = Math.min((speedRatio - 1) * 2, 1);
            const hue = 0 + intensity * 30; // Red to orange-yellow
            const saturation = 100;
            const lightness = 50 + intensity * 20;
            return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        } else {
            // Slow particles: cool colors (blue to cyan)
            const intensity = Math.min((1 - speedRatio) * 2, 1);
            const hue = 220 - intensity * 20; // Blue to cyan
            const saturation = 80 + intensity * 20;
            const lightness = 40 + intensity * 20;
            return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        }
    }
}

class Chamber {
    constructor(x, y, width, height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.particles = [];
        this.initialTemp = 300;
    }
    
    containsPoint(x, y) {
        return x >= this.x && x <= this.x + this.width &&
               y >= this.y && y <= this.y + this.height;
    }
    
    calculateTemperature() {
        if (this.particles.length === 0) return this.initialTemp;
        
        // For 2D ideal gas: average KE per particle = kT (not 3kT/2)
        // KE = (1/2) * m * v²
        // T = (total KE) / (N * k_B)
        
        let totalKE = 0;
        
        for (const particle of this.particles) {
            // Convert back to normalized units
            const vx_norm = particle.vx / VELOCITY_SCALE;
            const vy_norm = particle.vy / VELOCITY_SCALE;
            const speedSquaredNorm = vx_norm * vx_norm + vy_norm * vy_norm;
            totalKE += 0.5 * MASS * speedSquaredNorm;
        }
        
        // Temperature in Kelvin
        const temperature = (totalKE / (this.particles.length * k_B)) * T_REF;
        
        // Safeguard against unrealistic temperatures only
        return Math.max(1, Math.min(10000, temperature)); // Much higher upper limit
    }
}

class MaxwellDemonSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        
        // Collision effects
        this.collisionEffects = [];
        this.maxCollisionEffects = 50;
        
        // Track recently measured particles to avoid double-counting
        this.recentlyMeasured = new Set();
        this.measurementCooldown = 0.1; // seconds before particle can be measured again
        
        // Track particles passing through gate
        this.particlesPassedLeft = 0;
        this.particlesPassedRight = 0;
        
        this.gateWidth = GATE_WIDTH;
        this.gateOpen = false;
        
        this.leftChamber = new Chamber(0, 0, this.width / 2 - this.gateWidth / 2, this.height);
        this.rightChamber = new Chamber(this.width / 2 + this.gateWidth / 2, 0, 
                                       this.width / 2 - this.gateWidth / 2, this.height);
        
        this.particles = [];
        this.initialTemperature = 300;
        this.particleCount = 150;
        this.slowMotion = false;
        this.running = false;
        this.paused = false;
        this.speedMultiplier = 1;
        this.demonActive = true;
        
        this.startTime = 0;
        this.gateOperations = 0;
        this.gateOperationTime = Date.now();
        
        this.lastTime = 0;
        this.lastGateState = false;
        this.lastEnergyCheck = -1;
        this.initialEnergy = 0;
        this.successfulSorts = 0;
        this.totalGateCycles = 0;
        this.informationBits = 0;
        this.lastDemonActive = true;
        
        // Velocity histogram removed
        
        // Fluctuation tracking
        this.maxEntropyDecrease = 0;
        this.initialEntropy = 0;
        
        // Phase space visualization
        this.showPhaseSpace = true;
        this.phaseSpaceData = null;
        this.lastPhaseSpaceUpdate = 0;
        this.phaseSpaceUpdateInterval = 200; // Update every 200ms (less frequent)
        
        // Demon memory for realistic mode
        this.demonMode = 'perfect'; // 'perfect' or 'realistic'
        this.demonMemoryStrategy = 'fifo'; // Memory management strategy
        this.demonMemory = new DemonMemory(DEMON_MEMORY_LIMIT, this.demonMemoryStrategy);
        this.erasureCost = 0;
        this.entropyBreakdown = {
            chamber: 0,
            measurement: 0,
            erasure: 0,
            total: 0
        };
        
        // Data recording with circular buffer
        this.dataHistoryMaxSize = 5000; // Limit to prevent memory issues
        this.dataHistory = {
            time: [],
            tempLeft: [],
            tempRight: [],
            entropy: [],
            information: [],
            erasureCost: []
        };
        
        // Educational annotations
        this.showAnnotations = false;
        this.nextParticleToSort = null;
        
        // Performance monitoring for adaptive collision detection
        this.performanceMonitor = {
            frameTimeHistory: [],
            averageFrameTime: 0,
            useOptimizedCollisions: false,
            particleThreshold: 200, // Initial guess, will adapt
            lastSwitchTime: 0,
            targetFrameTime: 5 // ms, for 60 FPS with headroom
        };
        this.showPerformance = false;
        
        // Display smoothing and update control
        this.displayUpdateInterval = 1000; // Update display every 1 second (less frequent)
        this.lastDisplayUpdate = 0;
        this.smoothedValues = {
            leftTemp: 300,
            rightTemp: 300,
            entropyChange: 0,
            measurementCost: 0,
            netEntropy: 0,
            efficiency: 0
        };
        this.smoothingFactor = 0.1; // Balanced smoothing
    }
    
    init(temperature, particleCount) {
        this.initialTemperature = temperature;
        this.particleCount = particleCount;
        this.particles = [];
        this.leftChamber.particles = [];
        this.rightChamber.particles = [];
        
        // Distribute particles evenly between chambers
        const particlesPerChamber = Math.floor(particleCount / 2);
        
        // Set initial temperature for chambers
        this.leftChamber.initialTemp = temperature;
        this.rightChamber.initialTemp = temperature;
        
        // Left chamber particles
        for (let i = 0; i < particlesPerChamber; i++) {
            const x = this.leftChamber.x + Math.random() * this.leftChamber.width;
            const y = Math.random() * this.height;
            const particle = new Particle(x, y, temperature);
            this.particles.push(particle);
        }
        
        // Right chamber particles
        for (let i = 0; i < particleCount - particlesPerChamber; i++) {
            const x = this.rightChamber.x + Math.random() * this.rightChamber.width;
            const y = Math.random() * this.height;
            const particle = new Particle(x, y, temperature);
            this.particles.push(particle);
        }
        
        this.startTime = Date.now();
        this.gateOperations = 0;
        this.gateOperationTime = Date.now();
        
        // Store initial energy and entropy for verification
        this.initialEnergy = this.calculateTotalEnergy();
        this.initialEntropy = this.calculateEntropy();
        this.maxEntropyDecrease = 0;
        this.successfulSorts = 0;
        this.totalGateCycles = 0;
        this.informationBits = 0;
    }
    
    checkWallCollisions(particle) {
        const centerX = this.width / 2;
        const leftGateEdge = centerX - this.gateWidth / 2;
        const rightGateEdge = centerX + this.gateWidth / 2;
        
        // Wall collisions
        if (particle.x - particle.radius <= 0 && particle.vx < 0) {
            particle.vx = -particle.vx;
            particle.x = particle.radius;
        }
        
        if (particle.x + particle.radius >= this.width && particle.vx > 0) {
            particle.vx = -particle.vx;
            particle.x = this.width - particle.radius;
        }
        
        if (particle.y - particle.radius <= 0 && particle.vy < 0) {
            particle.vy = -particle.vy;
            particle.y = particle.radius;
        }
        
        if (particle.y + particle.radius >= this.height && particle.vy > 0) {
            particle.vy = -particle.vy;
            particle.y = this.height - particle.radius;
        }
        
        // Gate collision (when closed)
        if (!this.gateOpen) {
            const inGateX = particle.x > leftGateEdge - particle.radius && 
                           particle.x < rightGateEdge + particle.radius;
            
            if (inGateX) {
                if (particle.x < centerX && particle.vx > 0) {
                    particle.vx = -particle.vx;
                    particle.x = leftGateEdge - particle.radius;
                } else if (particle.x > centerX && particle.vx < 0) {
                    particle.vx = -particle.vx;
                    particle.x = rightGateEdge + particle.radius;
                }
            }
        }
    }
    
    updateChambers() {
        // Store previous chamber assignments
        const prevLeftCount = this.leftChamber.particles.length;
        const prevRightCount = this.rightChamber.particles.length;
        
        this.leftChamber.particles = [];
        this.rightChamber.particles = [];
        
        if (this.gateOpen && !this.demonActive) {
            // When gate is open and demon is off, treat as one chamber
            // Assign particles to maintain roughly equal distribution for temperature calculation
            const centerX = this.width / 2;
            for (const particle of this.particles) {
                if (particle.x < centerX) {
                    this.leftChamber.particles.push(particle);
                } else {
                    this.rightChamber.particles.push(particle);
                }
            }
        } else {
            // Normal chamber assignment when demon is active or gate is closed
            for (const particle of this.particles) {
                if (this.leftChamber.containsPoint(particle.x, particle.y)) {
                    this.leftChamber.particles.push(particle);
                } else if (this.rightChamber.containsPoint(particle.x, particle.y)) {
                    this.rightChamber.particles.push(particle);
                }
                // Particles in gate area are not assigned to either chamber when demon is active
            }
        }
        
        // Track changes in particle counts
        const leftChange = this.leftChamber.particles.length - prevLeftCount;
        const rightChange = this.rightChamber.particles.length - prevRightCount;
        
    }
    
    checkDemonLogic() {
        // If demon is inactive, keep gate open for free exchange
        if (!this.demonActive) {
            this.gateOpen = true;
            // When demon turns off, ensure energy is conserved perfectly
            // by re-checking all particle energies
            if (this.lastDemonActive && !this.demonActive) {
                this.enforceEnergyConservation();
            }
            this.lastDemonActive = false;
            return;
        }
        
        this.lastDemonActive = true;
        
        // Find approaching particles
        const predictions = this.findApproachingParticles();
        
        // Make gate decision
        const shouldOpen = this.makeGateDecision(predictions);
        
        // Record measurement only when making an actual gate decision
        // Perfect demon only needs to know fast/slow for the particle at the gate
        if (predictions.length > 0 && this.gateOpen !== shouldOpen) {
            // Only record when we're changing the gate state based on a measurement
            this.recordMeasurement(predictions[0]);
        }
        
        // Track state changes and information
        if (this.gateOpen !== shouldOpen) {
            this.handleGateStateChange(shouldOpen, predictions);
        }
        
        this.gateOpen = shouldOpen;
    }
    
    findApproachingParticles() {
        const theoreticalMeanSpeed = this.calculateTheoreticalMeanSpeed();
        const centerX = this.width / 2;
        const predictions = [];
        
        for (const particle of this.particles) {
            const onLeftSide = particle.x < centerX;
            const movingRight = particle.vx > 0;
            const movingLeft = particle.vx < 0;
            
            // Check if particle is approaching gate
            if ((onLeftSide && movingRight) || (!onLeftSide && movingLeft)) {
                const distToGate = Math.abs(particle.x - centerX);
                const timeToGate = distToGate / Math.abs(particle.vx);
                
                if (timeToGate < GATE_LOOK_AHEAD_TIME) {
                    // Check if we've recently measured this particle
                    const particleId = particle.x + '_' + particle.y; // Simple ID
                    const measurementKey = particleId + '_' + Math.floor(Date.now() / (this.measurementCooldown * 1000));
                    
                    if (!this.recentlyMeasured.has(measurementKey)) {
                        const evaluation = this.evaluateParticle(particle, timeToGate, theoreticalMeanSpeed);
                        predictions.push(evaluation);
                        
                        
                        // Don't record measurement yet - only record when gate decision is made
                        // This prevents over-counting measurements
                        this.recentlyMeasured.add(measurementKey);
                        
                        // Clean up old measurements periodically
                        if (this.recentlyMeasured.size > 1000) {
                            this.recentlyMeasured.clear();
                        }
                    }
                }
            }
        }
        
        // Sort by arrival time
        return predictions.sort((a, b) => a.timeToGate - b.timeToGate);
    }
    
    evaluateParticle(particle, timeToGate, meanSpeed) {
        const centerX = this.width / 2;
        const onLeftSide = particle.x < centerX;
        
        // Measure particle speed with noise in realistic mode
        let measuredSpeed = particle.speed;
        if (this.demonMode === 'realistic') {
            const noise = 1 + (Math.random() - 0.5) * 2 * MEASUREMENT_NOISE;
            measuredSpeed = particle.speed * noise;
        }
        
        const isFast = measuredSpeed > meanSpeed;
        const desirable = (isFast && onLeftSide) || (!isFast && !onLeftSide);
        
        return {
            particle,
            timeToGate,
            measuredSpeed,
            actualSpeed: particle.speed,
            isFast,
            desirable,
            measurementError: Math.abs(measuredSpeed - particle.speed) / particle.speed
        };
    }
    
    makeGateDecision(predictions) {
        if (predictions.length === 0) return false;
        
        // Open gate if the closest particle is desirable and reasonably close
        const closest = predictions[0];
        // Increased threshold to make demon more responsive
        const timeThreshold = this.demonMode === 'perfect' ? 0.1 : 0.08;
        return closest.desirable && closest.timeToGate < timeThreshold;
    }
    
    handleGateStateChange(shouldOpen, predictions) {
        this.gateOperations++;
        
        if (shouldOpen && predictions.length > 0 && predictions[0].desirable) {
            this.totalGateCycles++;
            
            // Only count as successful if measurement was correct
            // In perfect mode, always correct; in realistic mode, check if measurement matches reality
            const pred = predictions[0];
            const actuallyFast = pred.actualSpeed > this.calculateTheoreticalMeanSpeed();
            const measuredCorrectly = (pred.isFast === actuallyFast);
            
            if (measuredCorrectly) {
                this.successfulSorts++;
            }
            
            // Note: Measurement already recorded in findApproachingParticles
            // This ensures we account for ALL particles examined, not just sorted ones
            
            // Store for annotations
            if (this.showAnnotations) {
                this.nextParticleToSort = predictions[0].particle;
            }
        }
    }
    
    recordMeasurement(prediction) {
        // Perfect demon: exactly 1 bit per measurement (fast/slow)
        // Realistic demon: additional bits for speed value, position, etc.
        const bitsPerMeasurement = this.demonMode === 'perfect' ? 1 : 3;
        this.informationBits += bitsPerMeasurement;
        
        if (this.demonMode === 'realistic') {
            // Store measurement in finite memory using selected strategy
            const measurement = {
                particleId: prediction.particle,
                measuredSpeed: prediction.measuredSpeed,
                actualSpeed: prediction.actualSpeed,
                isFast: prediction.isFast,
                measurementError: prediction.measurementError
            };
            
            // Store and update erasure cost
            const storageCost = this.demonMemory.store(measurement);
            // In realistic mode, erasure is inefficient (1.5x to 3x theoretical minimum)
            const efficiencyFactor = 2.0; // 2x inefficiency for realistic demon
            this.erasureCost = this.demonMemory.getErasureCost(efficiencyFactor);
        }
    }
    
    
    calculateMeanSpeed() {
        const totalSpeed = this.particles.reduce((sum, p) => sum + p.speed, 0);
        return totalSpeed / this.particles.length;
    }
    
    calculateTheoreticalMeanSpeed() {
        // For 2D Maxwell-Boltzmann: <v> = sqrt(π*kT/2m)
        const avgTemp = (this.leftChamber.calculateTemperature() + 
                        this.rightChamber.calculateTemperature()) / 2;
        return Math.sqrt(Math.PI * k_B * avgTemp / (2 * MASS * T_REF)) * VELOCITY_SCALE;
    }
    
    calculateTotalEnergy() {
        // Total kinetic energy in normalized units
        return this.particles.reduce((sum, p) => {
            const vx_norm = p.vx / VELOCITY_SCALE;
            const vy_norm = p.vy / VELOCITY_SCALE;
            return sum + 0.5 * MASS * (vx_norm * vx_norm + vy_norm * vy_norm);
        }, 0);
    }
    
    enforceEnergyConservation() {
        // Rescale all velocities to match initial energy exactly
        const currentEnergy = this.calculateTotalEnergy();
        if (currentEnergy > 0 && this.initialEnergy > 0) {
            const scaleFactor = Math.sqrt(this.initialEnergy / currentEnergy);
            
            // Apply scaling to all particles
            for (const particle of this.particles) {
                particle.vx *= scaleFactor;
                particle.vy *= scaleFactor;
            }
        }
    }
    
    calculateEntropy() {
        // Calculate total entropy using Sackur-Tetrode equation for 2D ideal gas
        // S = N*k_B*[ln(A/(N*λ²)) + 1]
        // where λ² = h²/(2πmkT) is the thermal de Broglie wavelength squared
        
        // Update chamber particles
        this.updateChambers();
        
        const N = this.particles.length;
        const n_L = this.leftChamber.particles.length;
        const n_R = this.rightChamber.particles.length;
        
        // If all particles are in one chamber, use initial entropy as baseline
        if (n_L === 0 || n_R === 0) return this.initialEntropy;
        
        // When demon is OFF and gate is open, treat as single chamber for correct physics
        if (this.gateOpen && !this.demonActive) {
            // Calculate entropy as one unified system
            const totalArea = this.width * this.height; // Full area
            const avgTemp = this.calculateAverageTemperature();
            const lambda2 = H2_OVER_2PI_MK * T_REF / avgTemp;
            
            const arg = totalArea / (N * lambda2);
            if (arg > 0) {
                const S_unified = N * (Math.log(arg) + 1);
                const finalEntropy = k_B * S_unified;
                
                // Debug logging
                if (this.frameCount % 300 === 0) { // Less frequent logging
                    console.log('Unified chamber (demon OFF):');
                    console.log('  N:', N, 'T_avg:', avgTemp.toFixed(1), 'K');
                    console.log('  Total area:', totalArea);
                    console.log('  lambda2:', lambda2.toExponential(3));
                    console.log('  arg:', arg.toExponential(3));
                    console.log('  S_unified:', S_unified.toFixed(3));
                    console.log('  Final entropy:', finalEntropy.toFixed(3), 'k_B');
                }
                
                return finalEntropy;
            }
        }
        
        // When demon is active, calculate as separate chambers
        const T_L = this.leftChamber.calculateTemperature();
        const T_R = this.rightChamber.calculateTemperature();
        
        // Areas (in pixel² units)
        const A_L = this.leftChamber.width * this.leftChamber.height;
        const A_R = this.rightChamber.width * this.rightChamber.height;
        
        // Calculate thermal de Broglie wavelength squared for each chamber
        const lambda2_L = H2_OVER_2PI_MK * T_REF / T_L;
        const lambda2_R = H2_OVER_2PI_MK * T_REF / T_R;
        
        let S_total = 0;
        
        // Left chamber entropy: S = N*k_B*[ln(A/(N*λ²)) + 1]
        if (n_L > 0 && T_L > 0) {
            const arg_L = A_L / (n_L * lambda2_L);
            if (arg_L > 0) {
                const S_L = n_L * (Math.log(arg_L) + 1);
                S_total += S_L;
            }
        }
        
        // Right chamber entropy
        if (n_R > 0 && T_R > 0) {
            const arg_R = A_R / (n_R * lambda2_R);
            if (arg_R > 0) {
                const S_R = n_R * (Math.log(arg_R) + 1);
                S_total += S_R;
            }
        }
        
        const finalEntropy = k_B * S_total;
        
        // Debug logging for separated chambers
        if (this.frameCount % 300 === 0 && this.demonActive) {
            console.log('Separated chambers (demon ON):');
            console.log('  Left: n_L=', n_L, 'T_L=', T_L.toFixed(1), 'K');
            console.log('  Right: n_R=', n_R, 'T_R=', T_R.toFixed(1), 'K');
            console.log('  Final entropy:', finalEntropy.toFixed(3), 'k_B');
        }
        
        return finalEntropy;
    }
    
    calculateAverageTemperature() {
        // Calculate average temperature of all particles
        let totalKE = 0;
        for (const particle of this.particles) {
            const vx_norm = particle.vx / VELOCITY_SCALE;
            const vy_norm = particle.vy / VELOCITY_SCALE;
            const speedSquaredNorm = vx_norm * vx_norm + vy_norm * vy_norm;
            totalKE += 0.5 * MASS * speedSquaredNorm;
        }
        return (totalKE / (this.particles.length * k_B)) * T_REF;
    }
    
    calculateEntropyChange() {
        // Calculate entropy change from initial state
        const currentEntropy = this.calculateEntropy();
        const entropyChange = currentEntropy - this.initialEntropy;
        
        
        // Track maximum decrease
        const entropyDecrease = -entropyChange;
        this.maxEntropyDecrease = Math.max(this.maxEntropyDecrease, entropyDecrease);
        
        // Return percentage decrease in entropy
        const percentDecrease = this.initialEntropy > 0 
            ? (entropyDecrease / Math.abs(this.initialEntropy)) * 100 
            : 0;
        return Math.max(0, percentDecrease);
    }
    
    calculateFluctuationProbability() {
        // Calculate probability of current entropy decrease
        // According to fluctuation theorem: P(ΔS) = exp(-ΔS/k)
        const currentEntropy = this.calculateEntropy();
        const deltaS = this.initialEntropy - currentEntropy; // Positive for decrease
        
        // Probability in units where k_B = 1
        const probability = Math.exp(-deltaS);
        return probability;
    }
    
    checkParticleCollisions() {
        // Start timing collision detection
        const startTime = performance.now();
        
        // Use adaptive threshold based on performance
        if (this.performanceMonitor.useOptimizedCollisions || 
            this.particles.length > this.performanceMonitor.particleThreshold) {
            this.checkParticleCollisionsOptimized();
        } else {
            this.checkParticleCollisionsBasic();
        }
        
        // Update performance metrics
        const collisionTime = performance.now() - startTime;
        this.updatePerformanceMetrics(collisionTime);
    }
    
    checkParticleCollisionsBasic() {
        
        // Proper elastic collision detection between particles
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const p1 = this.particles[i];
                const p2 = this.particles[j];
                
                const dx = p2.x - p1.x;
                const dy = p2.y - p1.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const minDistance = p1.radius + p2.radius;
                
                if (distance < minDistance && distance > 0) {
                    // Normalize collision vector
                    const nx = dx / distance;
                    const ny = dy / distance;
                    
                    // Relative velocity
                    const dvx = p1.vx - p2.vx;
                    const dvy = p1.vy - p2.vy;
                    
                    // Relative velocity in collision normal direction
                    const dvn = dvx * nx + dvy * ny;
                    
                    // Do not resolve if velocities are separating
                    if (dvn < 0) continue;
                    
                    // Collision resolution for equal mass particles
                    // In 1D: v1' = v2, v2' = v1 (velocities exchange)
                    // In 2D: only normal components exchange
                    
                    // Project velocities onto collision normal and tangent
                    const v1n = p1.vx * nx + p1.vy * ny;
                    const v1t = p1.vx * (-ny) + p1.vy * nx;
                    const v2n = p2.vx * nx + p2.vy * ny;
                    const v2t = p2.vx * (-ny) + p2.vy * nx;
                    
                    // Exchange normal components (tangent components unchanged)
                    p1.vx = v2n * nx - v1t * ny;
                    p1.vy = v2n * ny + v1t * nx;
                    p2.vx = v1n * nx - v2t * ny;
                    p2.vy = v1n * ny + v2t * nx;
                    
                    // Add collision effect
                    const collisionX = (p1.x + p2.x) / 2;
                    const collisionY = (p1.y + p2.y) / 2;
                    const relativeSpeed = Math.sqrt(dvx * dvx + dvy * dvy) / VELOCITY_SCALE;
                    if (this.collisionEffects.length < this.maxCollisionEffects) {
                        this.collisionEffects.push(new CollisionEffect(collisionX, collisionY, relativeSpeed));
                    }
                    
                    // Separate particles to prevent overlap
                    // Only separate if they're still overlapping after velocity update
                    const newDx = p2.x - p1.x;
                    const newDy = p2.y - p1.y;
                    const newDistance = Math.sqrt(newDx * newDx + newDy * newDy);
                    
                    if (newDistance < minDistance) {
                        const overlap = minDistance - newDistance;
                        const correction = overlap / (2 * newDistance + 0.0001); // Avoid division by zero
                        const correctionX = newDx * correction;
                        const correctionY = newDy * correction;
                        
                        // Move particles apart symmetrically
                        p1.x -= correctionX;
                        p1.y -= correctionY;
                        p2.x += correctionX;
                        p2.y += correctionY;
                    }
                }
            }
        }
    }
    
    updatePerformanceMetrics(collisionTime) {
        const monitor = this.performanceMonitor;
        
        // Add to history (keep last 60 frames)
        monitor.frameTimeHistory.push(collisionTime);
        if (monitor.frameTimeHistory.length > 60) {
            monitor.frameTimeHistory.shift();
        }
        
        // Calculate rolling average
        monitor.averageFrameTime = monitor.frameTimeHistory.reduce((a, b) => a + b, 0) / monitor.frameTimeHistory.length;
        
        // Adaptive switching logic
        const now = Date.now();
        const timeSinceSwitch = now - monitor.lastSwitchTime;
        
        // Only switch methods every 1 second to avoid oscillation
        if (timeSinceSwitch > 1000) {
            if (!monitor.useOptimizedCollisions && monitor.averageFrameTime > monitor.targetFrameTime) {
                // Switch to optimized if frame time is too high
                monitor.useOptimizedCollisions = true;
                monitor.particleThreshold = Math.floor(this.particles.length * 0.9); // Set threshold below current count
                monitor.lastSwitchTime = now;
            } else if (monitor.useOptimizedCollisions && monitor.averageFrameTime < monitor.targetFrameTime * 0.5) {
                // Switch back to basic if we have lots of headroom
                monitor.useOptimizedCollisions = false;
                monitor.particleThreshold = Math.floor(this.particles.length * 1.1); // Set threshold above current count
                monitor.lastSwitchTime = now;
            }
        }
    }
    
    checkParticleCollisionsOptimized() {
        // Spatial hashing for O(n) collision detection
        const cellSize = COLLISION_CELL_SIZE; // Larger than particle diameter
        const grid = new Map();
        
        // Hash particles into grid cells
        for (const particle of this.particles) {
            const cellX = Math.floor(particle.x / cellSize);
            const cellY = Math.floor(particle.y / cellSize);
            const key = `${cellX},${cellY}`;
            
            if (!grid.has(key)) {
                grid.set(key, []);
            }
            grid.get(key).push(particle);
        }
        
        // Check collisions only within and adjacent cells
        for (const [key, cellParticles] of grid) {
            const [cellX, cellY] = key.split(',').map(Number);
            
            // Check within cell
            for (let i = 0; i < cellParticles.length; i++) {
                for (let j = i + 1; j < cellParticles.length; j++) {
                    this.resolveCollision(cellParticles[i], cellParticles[j]);
                }
            }
            
            // Check adjacent cells
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    if (dx === 0 && dy === 0) continue;
                    
                    const adjacentKey = `${cellX + dx},${cellY + dy}`;
                    if (grid.has(adjacentKey)) {
                        const adjacentParticles = grid.get(adjacentKey);
                        for (const p1 of cellParticles) {
                            for (const p2 of adjacentParticles) {
                                this.resolveCollision(p1, p2);
                            }
                        }
                    }
                }
            }
        }
    }
    
    resolveCollision(p1, p2) {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const minDistance = p1.radius + p2.radius;
        
        if (distance < minDistance && distance > 0) {
            // Same collision resolution as before
            const nx = dx / distance;
            const ny = dy / distance;
            
            const dvx = p1.vx - p2.vx;
            const dvy = p1.vy - p2.vy;
            const dvn = dvx * nx + dvy * ny;
            
            if (dvn < 0) return;
            
            const v1n = p1.vx * nx + p1.vy * ny;
            const v1t = p1.vx * (-ny) + p1.vy * nx;
            const v2n = p2.vx * nx + p2.vy * ny;
            const v2t = p2.vx * (-ny) + p2.vy * nx;
            
            p1.vx = v2n * nx - v1t * ny;
            p1.vy = v2n * ny + v1t * nx;
            p2.vx = v1n * nx - v2t * ny;
            p2.vy = v1n * ny + v2t * nx;
            
            // Add collision effect
            const collisionX = (p1.x + p2.x) / 2;
            const collisionY = (p1.y + p2.y) / 2;
            const relativeSpeed = Math.sqrt(dvx * dvx + dvy * dvy) / VELOCITY_SCALE;
            if (this.collisionEffects.length < this.maxCollisionEffects) {
                this.collisionEffects.push(new CollisionEffect(collisionX, collisionY, relativeSpeed));
            }
            
            // Separate particles
            const newDx = p2.x - p1.x;
            const newDy = p2.y - p1.y;
            const newDistance = Math.sqrt(newDx * newDx + newDy * newDy);
            
            if (newDistance < minDistance) {
                const overlap = minDistance - newDistance;
                const correction = overlap / (2 * newDistance + 0.0001);
                const correctionX = newDx * correction;
                const correctionY = newDy * correction;
                
                p1.x -= correctionX;
                p1.y -= correctionY;
                p2.x += correctionX;
                p2.y += correctionY;
            }
        }
    }
    
    update(deltaTime) {
        if (!this.running || this.paused) return;
        
        // Limit timestep for stability
        const dt = Math.min(deltaTime, 0.016) * this.speedMultiplier;
        const steps = Math.ceil(dt / MAX_TIMESTEP);
        const stepDt = dt / steps;
        
        // Multiple small steps for better accuracy
        for (let step = 0; step < steps; step++) {
            this.updatePhysicsStep(stepDt);
        }
        
        this.updateDisplays();
    }
    
    updatePhysicsStep(dt) {
        this.checkDemonLogic();
        
        // Update particle positions
        for (const particle of this.particles) {
            particle.update(dt, this.slowMotion);
            this.checkWallCollisions(particle);
        }
        
        // Check particle-particle collisions (optimize by checking every other frame)
        if (this.frameCount % 2 === 0) {
            this.checkParticleCollisions();
        }
        
        // Update collision effects
        this.collisionEffects = this.collisionEffects.filter(effect => effect.update(dt));
        
        this.updateChambers();
    }
    
    step() {
        if (!this.running) return;
        this.updatePhysicsStep(0.01);
        this.updateDisplays();
        this.render();
    }
    
    updateDisplays() {
        const currentTime = Date.now();
        const shouldUpdateDisplay = (currentTime - this.lastDisplayUpdate) > this.displayUpdateInterval;
        
        // Calculate current values based on demon state
        let leftTemp, rightTemp;
        
        if (this.gateOpen && !this.demonActive) {
            // When gate is open and demon is off, show equilibrating temperatures
            // Both chambers should approach the average temperature
            const avgTemp = this.calculateAverageTemperature();
            
            // Show actual chamber temperatures approaching equilibrium
            const actualLeftTemp = this.leftChamber.calculateTemperature();
            const actualRightTemp = this.rightChamber.calculateTemperature();
            
            // For display, show the temperatures moving toward equilibrium
            leftTemp = actualLeftTemp;
            rightTemp = actualRightTemp;
        } else {
            // Normal chamber temperatures when demon is active or gate is closed
            leftTemp = this.leftChamber.calculateTemperature();
            rightTemp = this.rightChamber.calculateTemperature();
        }
        
        // Apply exponential smoothing
        this.smoothedValues.leftTemp = this.smoothedValues.leftTemp * (1 - this.smoothingFactor) + 
                                       leftTemp * this.smoothingFactor;
        this.smoothedValues.rightTemp = this.smoothedValues.rightTemp * (1 - this.smoothingFactor) + 
                                        rightTemp * this.smoothingFactor;
        
        // Only update display at intervals
        if (shouldUpdateDisplay) {
            this.lastDisplayUpdate = currentTime;
            
            const displayLeftTemp = Math.round(this.smoothedValues.leftTemp);
            const displayRightTemp = Math.round(this.smoothedValues.rightTemp);
            const tempDiff = Math.abs(displayLeftTemp - displayRightTemp);
            
            // Debug logging
            if (this.frameCount % 300 === 0) { // Log every 5 seconds
                console.log('Temperature debug:');
                console.log('  Raw temps: Left =', leftTemp.toFixed(1), 'K, Right =', rightTemp.toFixed(1), 'K');
                console.log('  Smoothed: Left =', this.smoothedValues.leftTemp.toFixed(1), 'K, Right =', this.smoothedValues.rightTemp.toFixed(1), 'K');
                console.log('  Display: Left =', displayLeftTemp, 'K, Right =', displayRightTemp, 'K');
                console.log('  Demon active:', this.demonActive, 'Gate open:', this.gateOpen);
                if (this.gateOpen && !this.demonActive) {
                    console.log('  Expected: Temperatures should be converging toward equilibrium');
                }
            }
            
            // Update displays with stable values
            document.getElementById('tempLeft').textContent = displayLeftTemp || this.initialTemperature;
            document.getElementById('tempRight').textContent = displayRightTemp || this.initialTemperature;
            document.getElementById('tempDiff').textContent = Math.round(tempDiff);
        }
        
        // Calculate actual entropy change with smoothing
        const entropyDecrease = this.calculateEntropyChange();
        this.smoothedValues.entropyChange = this.smoothedValues.entropyChange * (1 - this.smoothingFactor * 0.5) + 
                                            entropyDecrease * this.smoothingFactor * 0.5;
        
        if (shouldUpdateDisplay) {
            document.getElementById('entropyViolation').textContent = this.smoothedValues.entropyChange.toFixed(1);
        }
        
        // Calculate elapsed time
        const elapsed = this.running ? (Date.now() - this.startTime) / 1000 : 0;
        
        // Record data for export with circular buffer
        if (this.running && elapsed > 0 && shouldUpdateDisplay) {
            // Add smoothed data for cleaner exports
            this.dataHistory.time.push(elapsed);
            this.dataHistory.tempLeft.push(Math.round(this.smoothedValues.leftTemp));
            this.dataHistory.tempRight.push(Math.round(this.smoothedValues.rightTemp));
            this.dataHistory.entropy.push(this.calculateEntropy());
            this.dataHistory.information.push(this.informationBits);
            this.dataHistory.erasureCost.push(this.erasureCost);
            
            // Maintain circular buffer size
            if (this.dataHistory.time.length > this.dataHistoryMaxSize) {
                this.dataHistory.time.shift();
                this.dataHistory.tempLeft.shift();
                this.dataHistory.tempRight.shift();
                this.dataHistory.entropy.shift();
                this.dataHistory.information.shift();
                this.dataHistory.erasureCost.shift();
            }
        }
        
        // Calculate gate operations per second
        const gateUpdateTime = Date.now();
        const timeDiff = (gateUpdateTime - this.gateOperationTime) / 1000;
        if (timeDiff > 1) {
            const opsPerSecond = this.gateOperations / timeDiff;
            document.getElementById('gateOps').textContent = Math.round(opsPerSecond);
            this.gateOperations = 0;
            this.gateOperationTime = gateUpdateTime;
        }
        
        // Update timer and energy conservation
        if (this.running) {
            const elapsed = (Date.now() - this.startTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor(elapsed % 60);
            document.getElementById('timer').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            // Energy conservation display with smoothing
            const currentEnergy = this.calculateTotalEnergy();
            const energyRatio = currentEnergy / this.initialEnergy;
            if (shouldUpdateDisplay) {
                document.getElementById('energyConservation').textContent = 
                    (energyRatio * 100).toFixed(1);
            }
            
            // Demon efficiency with smoothing
            const efficiency = this.totalGateCycles > 0 
                ? (this.successfulSorts / this.totalGateCycles * 100)
                : 0;
            this.smoothedValues.efficiency = this.smoothedValues.efficiency * (1 - this.smoothingFactor * 0.3) + 
                                             efficiency * this.smoothingFactor * 0.3;
            if (shouldUpdateDisplay) {
                document.getElementById('demonEfficiency').textContent = 
                    Math.round(this.smoothedValues.efficiency);
            }
            
            // Information and Landauer limit - update less frequently
            if (shouldUpdateDisplay) {
                document.getElementById('infoBits').textContent = this.informationBits;
                const landauerEnergy = this.informationBits * Math.log(2); // kT*ln(2) per bit
                document.getElementById('landauerEnergy').textContent = landauerEnergy.toFixed(1);
                
                // Erasure cost (for realistic demon)
                document.getElementById('erasureCost').textContent = this.erasureCost.toFixed(1);
            }
            
            // Net entropy change including information costs
            // Correct physics: measurement/storage immediately creates entropy
            const currentEntropy = this.calculateEntropy();
            const systemEntropyChange = currentEntropy - this.initialEntropy;
            
            // Debug entropy values (disabled for performance)
            if (false && this.frameCount % 300 === 0) {
                console.log('Entropy breakdown:');
                console.log('Initial entropy:', this.initialEntropy.toFixed(3), 'k_B');
                console.log('Current entropy:', currentEntropy.toFixed(3), 'k_B');
                console.log('System entropy change:', systemEntropyChange.toFixed(3), 'k_B');
                console.log('Information bits:', this.informationBits);
            }
            
            // Measurement cost depends on demon mode AND whether demon is active
            let measurementCost, gateOperationCost, erasureEntropy;
            
            if (!this.demonActive) {
                // When demon is OFF, no measurement or operation costs
                // System naturally equilibrates, increasing entropy
                measurementCost = 0;
                gateOperationCost = 0;
                erasureEntropy = 0;
            } else if (this.demonMode === 'perfect') {
                // Perfect demon operates at the theoretical Landauer limit
                // Each bit costs exactly kT*ln(2), no more
                measurementCost = this.informationBits * Math.log(2) * k_B;
                
                // No additional costs for perfect demon
                gateOperationCost = 0; // Frictionless, reversible gate
                erasureEntropy = 0; // Perfect memory management, no erasure needed
                
                // Perfect demon demonstrates the theoretical limit of reversible computation
            } else {
                // Realistic demon has multiple sources of inefficiency
                // Base measurement cost with inefficiency
                const baseCost = this.informationBits * Math.log(2) * k_B;
                const measurementEfficiency = 0.5; // 50% efficient = 2x cost
                measurementCost = baseCost / measurementEfficiency;
                
                // Gate operation cost: mechanical friction and imperfect switching
                gateOperationCost = this.totalGateCycles * 0.1 * Math.log(2) * k_B;
                
                // Erasure cost: Additional entropy when information is erased from memory
                erasureEntropy = this.erasureCost * k_B;
            }
            
            // Total entropy must increase (Second Law)
            // ΔS_total = ΔS_chambers + ΔS_measurement + ΔS_gate + ΔS_erasure ≥ 0
            const totalEntropyChange = systemEntropyChange + measurementCost + gateOperationCost + erasureEntropy;
            
            // Store breakdown for display
            this.entropyBreakdown = {
                chamber: systemEntropyChange,
                measurement: measurementCost,
                gateOperations: gateOperationCost,
                erasure: erasureEntropy,
                total: totalEntropyChange
            };
            
            // Debug the exact values being displayed
            if (this.frameCount % 300 === 0) {
                console.log('Entropy breakdown being displayed:');
                console.log('Chamber ΔS:', this.entropyBreakdown.chamber.toFixed(3), 'k_B');
                console.log('Measurement cost:', this.entropyBreakdown.measurement.toFixed(3), 'k_B');
                console.log('Total net entropy:', this.entropyBreakdown.total.toFixed(3), 'k_B');
            }
            
            // Apply smoothing to entropy values
            this.smoothedValues.netEntropy = this.smoothedValues.netEntropy * (1 - this.smoothingFactor * 0.3) + 
                                             totalEntropyChange * this.smoothingFactor * 0.3;
            this.smoothedValues.measurementCost = this.smoothedValues.measurementCost * (1 - this.smoothingFactor * 0.3) + 
                                                  measurementCost * this.smoothingFactor * 0.3;
            
            // Update physics breakdown display only at intervals
            if (shouldUpdateDisplay) {
                document.getElementById('netEntropy').textContent = this.smoothedValues.netEntropy.toFixed(3);
                
                if (this.entropyBreakdown) {
                    document.getElementById('chamberEntropy').textContent = this.entropyBreakdown.chamber.toFixed(3);
                    document.getElementById('measurementEntropy').textContent = this.smoothedValues.measurementCost.toFixed(3);
                    document.getElementById('gateOperationsEntropy').textContent = this.entropyBreakdown.gateOperations.toFixed(3);
                    document.getElementById('erasureEntropy').textContent = this.entropyBreakdown.erasure.toFixed(3);
                }
            }
            
            // Visual warning if Second Law would be violated
            const netEntropyElement = document.getElementById('netEntropy').parentElement;
            if (totalEntropyChange < -0.001) {
                netEntropyElement.style.color = '#ff4444';
                netEntropyElement.style.textShadow = '0 0 10px rgba(255, 68, 68, 0.8)';
            } else if (totalEntropyChange < 0.01) {
                netEntropyElement.style.color = '#ffaa44';
                netEntropyElement.style.textShadow = '0 0 10px rgba(255, 170, 68, 0.8)';
            } else {
                netEntropyElement.style.color = '#44ff44';
                netEntropyElement.style.textShadow = '0 0 10px rgba(68, 255, 68, 0.8)';
            }
            
            // Fluctuation probability - update less frequently
            if (shouldUpdateDisplay) {
                const fluctProb = this.calculateFluctuationProbability();
                document.getElementById('fluctProb').textContent = fluctProb.toExponential(2);
                
                // Max entropy decrease
                const maxEntropyPercent = this.initialEntropy > 0 
                    ? (this.maxEntropyDecrease / Math.abs(this.initialEntropy)) * 100 
                    : 0;
                document.getElementById('maxEntropy').textContent = maxEntropyPercent.toFixed(1);
            }
        }
    }
    
    render() {
        // Increment frame counter
        this.frameCount = (this.frameCount || 0) + 1;
        
        // Create gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, this.width, this.height);
        gradient.addColorStop(0, '#0a0e1a');
        gradient.addColorStop(0.3, '#0f0f1f');
        gradient.addColorStop(0.7, '#1a0a1a');
        gradient.addColorStop(1, '#1a0e0a');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // Add subtle animated grid (reduced density) - only update every few frames
        if (this.frameCount % 3 === 0) {
            this.ctx.save();
            this.ctx.strokeStyle = 'rgba(100, 100, 255, 0.02)';
            this.ctx.lineWidth = 1;
            const gridSize = 80;
            const gridOffset = (Date.now() / 400) % gridSize;
            
            this.ctx.beginPath();
            // Vertical lines (fewer)
            for (let x = -gridSize + gridOffset; x < this.width + gridSize; x += gridSize) {
                this.ctx.moveTo(x, 0);
                this.ctx.lineTo(x - 10, this.height);
            }
            
            // Horizontal lines (fewer)
            for (let y = -gridSize + gridOffset; y < this.height + gridSize; y += gridSize) {
                this.ctx.moveTo(0, y);
                this.ctx.lineTo(this.width, y - 5);
            }
            this.ctx.stroke();
            this.ctx.restore();
        }
        
        const centerX = this.width / 2;
        
        // Draw subtle chamber indicators
        this.ctx.fillStyle = 'rgba(52, 152, 219, 0.02)';
        this.ctx.fillRect(0, 0, centerX - this.gateWidth / 2, this.height);
        this.ctx.fillStyle = 'rgba(231, 76, 60, 0.02)';
        this.ctx.fillRect(centerX + this.gateWidth / 2, 0, this.width / 2, this.height);
        
        // Draw gate with enhanced effects
        if (this.gateOpen) {
            // Animated energy field effect for open gate
            this.ctx.save();
            
            // Pulsing glow
            const pulsePhase = (Date.now() / 1000) % 2;
            const pulseIntensity = 0.15 + Math.sin(pulsePhase * Math.PI) * 0.1;
            
            // Simplified glow effect
            const glowGradient = this.ctx.createLinearGradient(
                centerX - this.gateWidth / 2 - 10, 0,
                centerX + this.gateWidth / 2 + 10, 0
            );
            if (this.demonActive) {
                glowGradient.addColorStop(0, `rgba(100, 255, 150, 0)`);
                glowGradient.addColorStop(0.5, `rgba(100, 255, 150, ${pulseIntensity})`);
                glowGradient.addColorStop(1, `rgba(100, 255, 150, 0)`);
            } else {
                glowGradient.addColorStop(0, `rgba(150, 150, 200, 0)`);
                glowGradient.addColorStop(0.5, `rgba(150, 150, 200, ${pulseIntensity * 0.5})`);
                glowGradient.addColorStop(1, `rgba(150, 150, 200, 0)`);
            }
            this.ctx.fillStyle = glowGradient;
            this.ctx.fillRect(
                centerX - this.gateWidth / 2 - 10, 0, 
                this.gateWidth + 20, this.height
            );
            
            // Simplified energy field effect
            if (this.frameCount % 2 === 0) {  // Update every other frame
                this.ctx.strokeStyle = 'rgba(100, 255, 150, 0.3)';
                this.ctx.lineWidth = 1;
                const lineSpacing = 30;  // Fewer lines
                const animOffset = (this.frameCount / 2) % lineSpacing;
                
                this.ctx.beginPath();
                for (let y = -lineSpacing + animOffset; y < this.height + lineSpacing; y += lineSpacing) {
                    this.ctx.moveTo(centerX - this.gateWidth / 2, y);
                    this.ctx.lineTo(centerX + this.gateWidth / 2, y);
                }
                this.ctx.stroke();
            }
            
            this.ctx.restore();
        } else {
            // Simplified closed gate
            this.ctx.fillStyle = '#444';
            this.ctx.fillRect(centerX - this.gateWidth / 2, 0, this.gateWidth, this.height);
            
            // Simple hazard stripes (no animation for performance)
            this.ctx.save();
            this.ctx.strokeStyle = 'rgba(255, 100, 0, 0.4)';
            this.ctx.lineWidth = 3;
            
            this.ctx.beginPath();
            for (let y = 0; y < this.height; y += 40) {
                this.ctx.moveTo(centerX - this.gateWidth / 2, y);
                this.ctx.lineTo(centerX + this.gateWidth / 2, y + 20);
            }
            this.ctx.stroke();
            
            this.ctx.restore();
        }
        
        // Draw collision effects
        for (const effect of this.collisionEffects) {
            effect.render(this.ctx);
        }
        
        // Draw particles efficiently
        const meanSpeed = this.calculateMeanSpeed();
        
        // Draw trails first if in slow motion (batch operation)
        if (this.slowMotion) {
            this.ctx.save();
            this.ctx.lineWidth = 2;
            this.ctx.lineCap = 'round';
            
            for (const particle of this.particles) {
                if (particle.trail.length > 1) {
                    const speedRatio = particle.speed / meanSpeed;
                    const hue = speedRatio > 1 ? 0 : 220;
                    this.ctx.strokeStyle = `hsla(${hue}, 80%, 60%, 0.3)`;
                    
                    this.ctx.beginPath();
                    this.ctx.moveTo(particle.trail[0].x, particle.trail[0].y);
                    for (let i = 1; i < particle.trail.length; i++) {
                        this.ctx.lineTo(particle.trail[i].x, particle.trail[i].y);
                    }
                    this.ctx.stroke();
                }
            }
            this.ctx.restore();
        }
        
        // Draw all particles in single pass
        for (const particle of this.particles) {
            const speedRatio = particle.speed / meanSpeed;
            
            // Only add glow for extremely hot/cold particles (performance optimization)
            if (speedRatio > 2.0 || speedRatio < 0.5) {
                this.ctx.save();
                this.ctx.globalAlpha = 0.2;
                this.ctx.fillStyle = speedRatio > 2.0 ? '#ff6666' : '#6666ff';
                this.ctx.beginPath();
                this.ctx.arc(particle.x, particle.y, particle.radius * 2.5, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.restore();
            }
            
            // Main particle (no gradients for performance)
            this.ctx.fillStyle = particle.getColor(meanSpeed);
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // Draw demon at top with enhanced presence
        this.ctx.save();
        
        if (this.demonActive) {
            // Animated demon glow
            const glowPhase = (Date.now() / 500) % (Math.PI * 2);
            const glowSize = 30 + Math.sin(glowPhase) * 5;
            
            // Demon aura
            const auraGradient = this.ctx.createRadialGradient(
                centerX, 30, 0,
                centerX, 30, glowSize
            );
            auraGradient.addColorStop(0, 'rgba(255, 215, 0, 0.3)');
            auraGradient.addColorStop(0.5, 'rgba(255, 100, 0, 0.2)');
            auraGradient.addColorStop(1, 'rgba(255, 50, 0, 0)');
            this.ctx.fillStyle = auraGradient;
            this.ctx.beginPath();
            this.ctx.arc(centerX, 30, glowSize, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Demon emoji with shadow
            this.ctx.shadowColor = 'rgba(255, 100, 0, 0.5)';
            this.ctx.shadowBlur = 10;
            this.ctx.font = 'bold 36px monospace';
            this.ctx.fillStyle = '#ffd700';
            const demonY = 40 + Math.sin(glowPhase * 2) * 2; // Slight floating animation
            this.ctx.fillText('👹', centerX - 18, demonY);
            this.ctx.shadowBlur = 0;
            
            // Gate status indicator with glow
            this.ctx.font = 'bold 14px monospace';
            if (this.gateOpen) {
                this.ctx.shadowColor = 'rgba(0, 255, 0, 0.5)';
                this.ctx.shadowBlur = 5;
                this.ctx.fillStyle = '#00ff00';
                this.ctx.fillText('▼ OPEN ▼', centerX - 35, 65);
            } else {
                this.ctx.shadowColor = 'rgba(255, 0, 0, 0.5)';
                this.ctx.shadowBlur = 5;
                this.ctx.fillStyle = '#ff0000';
                this.ctx.fillText('◆ CLOSED ◆', centerX - 40, 65);
            }
            this.ctx.shadowBlur = 0;
            
            // Show theoretical mean speed threshold
            const theoreticalMean = this.calculateTheoreticalMeanSpeed();
            const actualMean = this.calculateMeanSpeed();
            this.ctx.font = '10px monospace';
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
            this.ctx.fillText(`Threshold: ${theoreticalMean.toFixed(0)} px/s`, centerX - 50, 85);
            this.ctx.fillText(`Current: ${actualMean.toFixed(0)} px/s`, centerX - 50, 98);
        } else {
            // Sleeping demon
            this.ctx.font = '32px monospace';
            this.ctx.fillStyle = '#666';
            const sleepY = 40 + Math.sin(Date.now() / 2000) * 3; // Gentle breathing animation
            this.ctx.fillText('😴', centerX - 16, sleepY);
            
            // Z's for sleeping
            const zPhase = (Date.now() / 1000) % 3;
            this.ctx.font = `${12 + zPhase * 2}px monospace`;
            this.ctx.fillStyle = `rgba(150, 150, 150, ${0.6 - zPhase * 0.2})`;
            this.ctx.fillText('z', centerX + 20, 30 - zPhase * 5);
            
            // Demon inactive indicator
            this.ctx.font = '12px monospace';
            this.ctx.fillStyle = 'rgba(150, 150, 150, 0.8)';
            this.ctx.fillText('SLEEPING', centerX - 30, 65);
        }
        
        this.ctx.restore();
        
        // Velocity histogram removed for simplicity
        
        // Draw phase space plot (always draw to prevent flickering)
        if (this.showPhaseSpace) {
            this.drawPhaseSpace();
        }
        
        // Draw educational annotations
        if (this.showAnnotations && this.nextParticleToSort) {
            this.drawAnnotations();
        }
        
        // Draw performance metrics
        if (this.showPerformance) {
            this.drawPerformanceMetrics();
        }
        
        // Draw memory visualization for realistic demon
        if (this.demonMode === 'realistic') {
            this.drawMemoryVisualization();
        }
    }
    
    
    drawPhaseSpace() {
        // Position and size
        const phaseX = this.width - PHASE_SPACE_X_OFFSET;
        const phaseY = this.height - PHASE_SPACE_Y_OFFSET;
        const phaseWidth = PHASE_SPACE_WIDTH;
        const phaseHeight = PHASE_SPACE_HEIGHT;
        
        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(phaseX - 10, phaseY - 10, phaseWidth + 20, phaseHeight + 20);
        
        // Update phase space data only at intervals
        const currentTime = Date.now();
        if (!this.phaseSpaceData || (currentTime - this.lastPhaseSpaceUpdate) > this.phaseSpaceUpdateInterval) {
            this.lastPhaseSpaceUpdate = currentTime;
            
            // Create density map for better visualization
            const gridSize = 10;
            const densityGrid = this.createDensityGrid(gridSize);
            
            this.phaseSpaceData = {
                densityGrid: densityGrid,
                gridSize: gridSize,
                maxDensity: Math.max(...densityGrid.flat())
            };
        }
        
        // Use cached data
        const { densityGrid, gridSize, maxDensity } = this.phaseSpaceData;
        if (maxDensity > 0) {
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const density = densityGrid[i][j];
                    if (density > 0) {
                        const intensity = density / maxDensity;
                        const cellX = phaseX + (i / gridSize) * phaseWidth;
                        const cellY = phaseY + (j / gridSize) * phaseHeight;
                        const cellW = phaseWidth / gridSize;
                        const cellH = phaseHeight / gridSize;
                        
                        // Color based on average speed in cell
                        const hue = intensity > 0.5 ? 0 : 240;
                        const alpha = 0.2 + intensity * 0.6;
                        this.ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${alpha})`;
                        this.ctx.fillRect(cellX, cellY, cellW, cellH);
                    }
                }
            }
        }
        
        // Draw axes
        this.ctx.strokeStyle = '#666';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(phaseX, phaseY + phaseHeight/2);
        this.ctx.lineTo(phaseX + phaseWidth, phaseY + phaseHeight/2);
        this.ctx.moveTo(phaseX + phaseWidth/2, phaseY);
        this.ctx.lineTo(phaseX + phaseWidth/2, phaseY + phaseHeight);
        this.ctx.stroke();
        
        // Labels
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '12px monospace';
        this.ctx.fillText('Phase Space (x vs vx)', phaseX, phaseY - 15);
        
        // Axis labels
        this.ctx.font = '10px monospace';
        this.ctx.fillStyle = '#888';
        this.ctx.fillText('Left', phaseX + 5, phaseY + phaseHeight/2 - 5);
        this.ctx.fillText('Right', phaseX + phaseWidth - 30, phaseY + phaseHeight/2 - 5);
        this.ctx.fillText('+vx', phaseX + phaseWidth/2 + 5, phaseY + 10);
        this.ctx.fillText('-vx', phaseX + phaseWidth/2 + 5, phaseY + phaseHeight - 5);
        
        // Add legend
        this.ctx.font = '9px monospace';
        this.ctx.fillStyle = '#f66';
        this.ctx.fillText('■ High density', phaseX + phaseWidth - 70, phaseY + 20);
        this.ctx.fillStyle = '#66f';
        this.ctx.fillText('■ Low density', phaseX + phaseWidth - 70, phaseY + 30);
    }
    
    createDensityGrid(gridSize) {
        const grid = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0));
        const centerX = this.width / 2;
        const maxVx = Math.max(...this.particles.map(p => Math.abs(p.vx)), 100);
        
        this.particles.forEach(p => {
            // Map to grid coordinates
            const relX = (p.x - centerX) / (this.width / 2); // -1 to 1
            const relVx = p.vx / maxVx; // -1 to 1
            
            const gridX = Math.floor((relX + 1) * 0.5 * gridSize);
            const gridY = Math.floor((1 - relVx) * 0.5 * gridSize);
            
            if (gridX >= 0 && gridX < gridSize && gridY >= 0 && gridY < gridSize) {
                grid[gridX][gridY]++;
            }
        });
        
        return grid;
    }
    
    drawAnnotations() {
        if (!this.nextParticleToSort) return;
        
        const particle = this.nextParticleToSort;
        const centerX = this.width / 2;
        
        // Draw arrow pointing to particle
        this.ctx.strokeStyle = '#ffff00';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        
        // Arrow from top
        const arrowStartY = 100;
        this.ctx.beginPath();
        this.ctx.moveTo(particle.x, arrowStartY);
        this.ctx.lineTo(particle.x, particle.y - particle.radius - 10);
        this.ctx.stroke();
        
        // Arrowhead
        this.ctx.beginPath();
        this.ctx.moveTo(particle.x, particle.y - particle.radius - 10);
        this.ctx.lineTo(particle.x - 5, particle.y - particle.radius - 15);
        this.ctx.lineTo(particle.x + 5, particle.y - particle.radius - 15);
        this.ctx.closePath();
        this.ctx.fillStyle = '#ffff00';
        this.ctx.fill();
        
        this.ctx.setLineDash([]);
        
        // Annotation text
        const isFast = particle.speed > this.calculateTheoreticalMeanSpeed();
        const onLeftSide = particle.x < centerX;
        const movingRight = particle.vx > 0;
        
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(particle.x - 80, arrowStartY - 40, 160, 35);
        
        this.ctx.fillStyle = '#ffff00';
        this.ctx.font = '11px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            `${isFast ? 'Fast' : 'Slow'} particle`,
            particle.x, arrowStartY - 25
        );
        this.ctx.fillText(
            `Gate: ${(isFast && onLeftSide && movingRight) || (!isFast && !onLeftSide && !movingRight) ? 'OPEN' : 'CLOSED'}`,
            particle.x, arrowStartY - 10
        );
        this.ctx.textAlign = 'left';
    }
    
    drawPerformanceMetrics() {
        const monitor = this.performanceMonitor;
        const x = this.width - 250;
        const y = 20;
        
        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(x - 10, y - 10, 240, 120);
        
        // Title
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 12px monospace';
        this.ctx.fillText('Performance Metrics', x, y + 5);
        
        // Metrics
        this.ctx.font = '11px monospace';
        this.ctx.fillStyle = '#aaa';
        
        const method = monitor.useOptimizedCollisions ? 'Spatial Hash' : 'Brute Force';
        const methodColor = monitor.useOptimizedCollisions ? '#4ecdc4' : '#ffd700';
        
        this.ctx.fillText('Collision Method:', x, y + 25);
        this.ctx.fillStyle = methodColor;
        this.ctx.fillText(method, x + 120, y + 25);
        
        this.ctx.fillStyle = '#aaa';
        this.ctx.fillText('Avg Frame Time:', x, y + 45);
        const frameTimeColor = monitor.averageFrameTime > monitor.targetFrameTime ? '#ff6b6b' : '#96ceb4';
        this.ctx.fillStyle = frameTimeColor;
        this.ctx.fillText(`${monitor.averageFrameTime.toFixed(2)} ms`, x + 120, y + 45);
        
        this.ctx.fillStyle = '#aaa';
        this.ctx.fillText('Particle Limit:', x, y + 65);
        this.ctx.fillStyle = '#fff';
        this.ctx.fillText(monitor.particleThreshold.toString(), x + 120, y + 65);
        
        this.ctx.fillText('Current Count:', x, y + 85);
        this.ctx.fillText(this.particles.length.toString(), x + 120, y + 85);
        
        // Frame time graph
        if (monitor.frameTimeHistory.length > 1) {
            const graphX = x;
            const graphY = y + 95;
            const graphWidth = 220;
            const graphHeight = 15;
            
            this.ctx.strokeStyle = '#333';
            this.ctx.strokeRect(graphX, graphY, graphWidth, graphHeight);
            
            this.ctx.strokeStyle = frameTimeColor;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            
            for (let i = 0; i < monitor.frameTimeHistory.length; i++) {
                const xPos = graphX + (i / monitor.frameTimeHistory.length) * graphWidth;
                const yPos = graphY + graphHeight - (monitor.frameTimeHistory[i] / 10) * graphHeight;
                
                if (i === 0) {
                    this.ctx.moveTo(xPos, yPos);
                } else {
                    this.ctx.lineTo(xPos, yPos);
                }
            }
            this.ctx.stroke();
            
            // Target line
            this.ctx.strokeStyle = '#ff0';
            this.ctx.setLineDash([2, 2]);
            this.ctx.beginPath();
            const targetY = graphY + graphHeight - (monitor.targetFrameTime / 10) * graphHeight;
            this.ctx.moveTo(graphX, targetY);
            this.ctx.lineTo(graphX + graphWidth, targetY);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
    }
    
    drawMemoryVisualization() {
        const stats = this.demonMemory.getStatistics();
        const x = 20;
        const y = 20;
        const width = 200;
        const height = 120;
        
        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(x - 10, y - 10, width + 20, height + 20);
        
        // Title
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 12px monospace';
        this.ctx.fillText('Demon Memory', x, y + 5);
        
        // Memory strategy
        this.ctx.font = '10px monospace';
        this.ctx.fillStyle = '#ffd700';
        const strategyNames = {
            'fifo': 'FIFO',
            'lru': 'LRU',
            'selective': 'Selective',
            'compressed': 'Compressed'
        };
        this.ctx.fillText(strategyNames[stats.strategy], x + 120, y + 5);
        
        // Memory utilization bar
        const barY = y + 25;
        const barHeight = 20;
        const utilization = stats.utilization;
        
        // Background bar
        this.ctx.fillStyle = '#333';
        this.ctx.fillRect(x, barY, width, barHeight);
        
        // Fill bar with gradient
        const fillWidth = width * utilization;
        const gradient = this.ctx.createLinearGradient(x, 0, x + width, 0);
        if (utilization < 0.7) {
            gradient.addColorStop(0, '#4ecdc4');
            gradient.addColorStop(1, '#45b7d1');
        } else if (utilization < 0.9) {
            gradient.addColorStop(0, '#f39c12');
            gradient.addColorStop(1, '#e67e22');
        } else {
            gradient.addColorStop(0, '#e74c3c');
            gradient.addColorStop(1, '#c0392b');
        }
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(x, barY, fillWidth, barHeight);
        
        // Utilization text
        this.ctx.fillStyle = '#fff';
        this.ctx.font = 'bold 11px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${Math.round(utilization * 100)}%`, x + width/2, barY + barHeight/2 + 4);
        this.ctx.textAlign = 'left';
        
        // Statistics
        this.ctx.font = '10px monospace';
        this.ctx.fillStyle = '#aaa';
        
        let statsY = barY + barHeight + 15;
        this.ctx.fillText(`Stored: ${stats.stored}/${stats.capacity} bits`, x, statsY);
        
        statsY += 15;
        this.ctx.fillText(`Erasures: ${stats.erasures}`, x, statsY);
        
        statsY += 15;
        this.ctx.fillText(`Storage Cost: ${stats.storageCost.toFixed(1)} bits`, x, statsY);
        
        statsY += 15;
        this.ctx.fillText(`Erasure Cost: ${stats.erasureCost.toFixed(2)} kT`, x, statsY);
        
        // Memory cells visualization (for compressed strategy)
        if (stats.strategy === 'compressed' && this.demonMemory.storage.length > 0) {
            const cellY = statsY + 20;
            const cellSize = 8;
            const cellGap = 2;
            const maxCells = Math.floor(width / (cellSize + cellGap));
            const cellsToShow = Math.min(this.demonMemory.storage.length, maxCells);
            
            for (let i = 0; i < cellsToShow; i++) {
                const entry = this.demonMemory.storage[i];
                const cellX = x + i * (cellSize + cellGap);
                
                if (entry.compressed) {
                    this.ctx.fillStyle = '#ff6b6b';
                    this.ctx.fillRect(cellX, cellY, cellSize/2, cellSize);
                } else {
                    this.ctx.fillStyle = '#4ecdc4';
                    this.ctx.fillRect(cellX, cellY, cellSize, cellSize);
                }
            }
            
            // Legend
            this.ctx.font = '9px monospace';
            this.ctx.fillStyle = '#4ecdc4';
            this.ctx.fillText('■ Full', x, cellY + cellSize + 12);
            this.ctx.fillStyle = '#ff6b6b';
            this.ctx.fillText('■ Compressed', x + 50, cellY + cellSize + 12);
        }
    }
    
    animate(currentTime) {
        if (!this.running) return;
        
        if (!this.lastTime) {
            this.lastTime = currentTime;
        }
        
        const deltaTime = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;
        
        this.update(deltaTime);
        this.render();
        
        requestAnimationFrame(this.animate.bind(this));
    }
    
    start() {
        this.running = true;
        this.paused = false;
        this.lastTime = 0;
        requestAnimationFrame(this.animate.bind(this));
    }
    
    pause() {
        this.paused = true;
    }
    
    resume() {
        this.paused = false;
        this.lastTime = performance.now();
    }
    
    stop() {
        this.running = false;
        this.paused = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('simulationCanvas');
    let simulation = null;
    
    const resizeCanvas = () => {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        if (simulation) {
            simulation.width = canvas.width;
            simulation.height = canvas.height;
            simulation.leftChamber = new Chamber(0, 0, canvas.width / 2 - simulation.gateWidth / 2, canvas.height);
            simulation.rightChamber = new Chamber(canvas.width / 2 + simulation.gateWidth / 2, 0, 
                                                 canvas.width / 2 - simulation.gateWidth / 2, canvas.height);
            simulation.render();
        }
    };
    
    // Set initial canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Create simulation after canvas is sized
    simulation = new MaxwellDemonSimulation(canvas);
    
    window.addEventListener('resize', resizeCanvas);

    document.getElementById('tempSlider').addEventListener('input', (e) => {
        document.getElementById('tempValue').textContent = e.target.value;
    });

    document.getElementById('particleCount').addEventListener('input', (e) => {
        document.getElementById('particleValue').textContent = e.target.value;
    });

    document.getElementById('speedMultiplier').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        document.getElementById('speedValue').textContent = value.toFixed(1);
        simulation.speedMultiplier = value;
    });

    document.getElementById('slowMotion').addEventListener('change', (e) => {
        simulation.slowMotion = e.target.checked;
    });

    document.getElementById('demonActive').addEventListener('change', (e) => {
        simulation.demonActive = e.target.checked;
        if (!e.target.checked) {
            // When demon turns off, ensure perfect energy conservation
            simulation.enforceEnergyConservation();
        }
    });

    document.getElementById('showHistogram').addEventListener('change', (e) => {
        simulation.showHistogram = e.target.checked;
    });

    document.getElementById('showPhaseSpace').addEventListener('change', (e) => {
        simulation.showPhaseSpace = e.target.checked;
    });

    document.getElementById('showAnnotations').addEventListener('change', (e) => {
        simulation.showAnnotations = e.target.checked;
    });

    document.getElementById('showPerformance').addEventListener('change', (e) => {
        simulation.showPerformance = e.target.checked;
    });

    document.getElementById('demonMode').addEventListener('change', (e) => {
        simulation.demonMode = e.target.value;
        
        // Show/hide memory strategy controls
        const showMemoryControls = e.target.value === 'realistic';
        document.getElementById('memoryStrategyGroup').style.display = showMemoryControls ? 'block' : 'none';
        document.getElementById('memoryCapacityGroup').style.display = showMemoryControls ? 'block' : 'none';
        
        // Reset memory when changing modes
        const capacity = parseInt(document.getElementById('memoryCapacity').value);
        const strategy = document.getElementById('memoryStrategy').value;
        simulation.demonMemory = new DemonMemory(capacity, strategy);
        simulation.erasureCost = 0;
    });

    document.getElementById('memoryStrategy').addEventListener('change', (e) => {
        simulation.demonMemoryStrategy = e.target.value;
        const capacity = parseInt(document.getElementById('memoryCapacity').value);
        simulation.demonMemory = new DemonMemory(capacity, e.target.value);
        simulation.erasureCost = 0;
    });

    document.getElementById('memoryCapacity').addEventListener('input', (e) => {
        const capacity = parseInt(e.target.value);
        document.getElementById('memoryCapacityValue').textContent = capacity;
        simulation.demonMemory = new DemonMemory(capacity, simulation.demonMemoryStrategy);
        simulation.erasureCost = 0;
    });

    document.getElementById('exportBtn').addEventListener('click', () => {
        exportSimulationData(simulation);
    });

    document.getElementById('launchBtn').addEventListener('click', () => {
        const temperature = parseInt(document.getElementById('tempSlider').value);
        const particleCount = parseInt(document.getElementById('particleCount').value);
        simulation.init(temperature, particleCount);
        simulation.start();
        
        // Update button states
        document.getElementById('launchBtn').disabled = true;
        document.getElementById('pauseBtn').disabled = false;
        document.getElementById('stepBtn').disabled = false;
    });

    document.getElementById('pauseBtn').addEventListener('click', () => {
        if (simulation.paused) {
            simulation.resume();
            document.getElementById('pauseBtn').textContent = 'Pause';
        } else {
            simulation.pause();
            document.getElementById('pauseBtn').textContent = 'Resume';
        }
    });

    document.getElementById('stepBtn').addEventListener('click', () => {
        if (!simulation.paused) {
            simulation.pause();
            document.getElementById('pauseBtn').textContent = 'Resume';
        }
        simulation.step();
    });

    document.getElementById('resetBtn').addEventListener('click', () => {
        simulation.stop();
        simulation.ctx.fillStyle = '#0a0a0a';
        simulation.ctx.fillRect(0, 0, simulation.width, simulation.height);
        document.getElementById('tempLeft').textContent = '300';
        document.getElementById('tempRight').textContent = '300';
        document.getElementById('tempDiff').textContent = '0';
        document.getElementById('timer').textContent = '00:00';
        document.getElementById('entropyViolation').textContent = '0';
        document.getElementById('gateOps').textContent = '0';
        document.getElementById('energyConservation').textContent = '100.0';
        document.getElementById('demonEfficiency').textContent = '0';
        document.getElementById('infoBits').textContent = '0';
        document.getElementById('landauerEnergy').textContent = '0';
        document.getElementById('erasureCost').textContent = '0';
        document.getElementById('netEntropy').textContent = '0';
        document.getElementById('fluctProb').textContent = '1.0';
        document.getElementById('maxEntropy').textContent = '0';
        
        // Reset data history
        simulation.dataHistory = {
            time: [],
            tempLeft: [],
            tempRight: [],
            entropy: [],
            information: [],
            erasureCost: []
        };
        
        // Reset demon memory with current settings
        const capacity = parseInt(document.getElementById('memoryCapacity').value);
        const strategy = document.getElementById('memoryStrategy').value;
        simulation.demonMemory = new DemonMemory(capacity, strategy);
        simulation.erasureCost = 0;
        
        // Reset button states
        document.getElementById('launchBtn').disabled = false;
        document.getElementById('pauseBtn').disabled = true;
        document.getElementById('pauseBtn').textContent = 'Pause';
        document.getElementById('stepBtn').disabled = true;
    });

    simulation.render();
    
    // Export function
    function exportSimulationData(sim) {
        const data = {
            metadata: {
                initialTemperature: sim.initialTemperature,
                particleCount: sim.particles.length,
                demonMode: sim.demonMode,
                exportTime: new Date().toISOString()
            },
            thermodynamics: {
                initialEntropy: sim.initialEntropy,
                currentEntropy: sim.calculateEntropy(),
                informationBits: sim.informationBits,
                erasureCost: sim.erasureCost,
                netEntropyChange: sim.calculateEntropy() - sim.initialEntropy + 
                                 sim.informationBits * Math.log(2) - sim.erasureCost
            },
            timeSeries: {
                time: sim.dataHistory.time,
                temperatureLeft: sim.dataHistory.tempLeft,
                temperatureRight: sim.dataHistory.tempRight,
                entropy: sim.dataHistory.entropy,
                information: sim.dataHistory.information,
                erasureCost: sim.dataHistory.erasureCost
            }
        };
        
        // Create downloadable JSON file
        const jsonStr = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `maxwell_demon_data_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});