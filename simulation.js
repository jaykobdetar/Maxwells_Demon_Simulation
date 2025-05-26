// Physical constants and unit system
// Using normalized units: distance in pixels, time in seconds
// kT = 1 at T = 300K, mass = 1
const k_B = 1.0; // Normalized Boltzmann constant
const T_REF = 300; // Reference temperature in Kelvin
const MASS = 1.0; // Particle mass in normalized units

// For entropy calculations, we need Planck's constant in our unit system
// h² / (2πmk) has dimensions of length² × temperature
// At T_REF = 300K, for argon atoms (m ≈ 6.6e-26 kg):
// λ_thermal = h/√(2πmkT) ≈ 1.6e-11 m
// In our normalized units where typical velocities are ~100 px/s
// and k_B = 1, we set this constant to give physically meaningful entropy
const H2_OVER_2PI_MK = 1e-6; // Normalized h²/(2πmk) to give proper entropy scale

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
    
    getErasureCost() {
        // Each erasure costs ln(2) in normalized units
        return this.erasureCount * Math.log(2);
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
        const hue = speedRatio > 1 ? 0 : 240;
        const intensity = Math.min(speedRatio > 1 ? speedRatio - 1 : 1 - speedRatio, 1);
        const lightness = 50 + intensity * 30;
        return `hsl(${hue}, 100%, ${lightness}%)`;
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
        
        return temperature;
    }
}

class MaxwellDemonSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        
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
        
        // For velocity distribution histogram
        this.showHistogram = true;
        this.histogramBins = HISTOGRAM_BINS;
        this.maxHistogramSpeed = 0;
        
        // Fluctuation tracking
        this.maxEntropyDecrease = 0;
        this.initialEntropy = 0;
        
        // Phase space visualization
        this.showPhaseSpace = true;
        
        // Demon memory for realistic mode
        this.demonMode = 'perfect'; // 'perfect' or 'realistic'
        this.demonMemoryStrategy = 'fifo'; // Memory management strategy
        this.demonMemory = new DemonMemory(DEMON_MEMORY_LIMIT, this.demonMemoryStrategy);
        this.erasureCost = 0;
        
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
        this.leftChamber.particles = [];
        this.rightChamber.particles = [];
        
        for (const particle of this.particles) {
            if (this.leftChamber.containsPoint(particle.x, particle.y)) {
                this.leftChamber.particles.push(particle);
            } else if (this.rightChamber.containsPoint(particle.x, particle.y)) {
                this.rightChamber.particles.push(particle);
            }
        }
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
                    predictions.push(this.evaluateParticle(particle, timeToGate, theoreticalMeanSpeed));
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
        
        // Open gate if the closest particle is desirable and very close
        const closest = predictions[0];
        return closest.desirable && closest.timeToGate < 0.05;
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
            
            // Record measurement
            this.recordMeasurement(predictions[0]);
            
            // Store for annotations
            if (this.showAnnotations) {
                this.nextParticleToSort = predictions[0].particle;
            }
        }
    }
    
    recordMeasurement(prediction) {
        // Each measurement costs 1 bit of information
        this.informationBits += 1;
        
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
            this.erasureCost = this.demonMemory.getErasureCost();
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
        
        if (n_L === 0 || n_R === 0) return 0; // Avoid log(0)
        
        // Get temperatures
        const T_L = this.leftChamber.calculateTemperature();
        const T_R = this.rightChamber.calculateTemperature();
        
        // Areas (in pixel² units)
        const A_L = this.leftChamber.width * this.leftChamber.height;
        const A_R = this.rightChamber.width * this.rightChamber.height;
        
        // Calculate thermal de Broglie wavelength squared for each chamber
        // λ² = h²/(2πmkT)
        const lambda2_L = H2_OVER_2PI_MK * T_REF / T_L;
        const lambda2_R = H2_OVER_2PI_MK * T_REF / T_R;
        
        let S_total = 0;
        
        // Left chamber entropy: S = N*k_B*[ln(A/(N*λ²)) + 1]
        if (n_L > 0 && T_L > 0) {
            const arg_L = A_L / (n_L * lambda2_L);
            if (arg_L > 0) {
                S_total += n_L * (Math.log(arg_L) + 1);
            }
        }
        
        // Right chamber entropy
        if (n_R > 0 && T_R > 0) {
            const arg_R = A_R / (n_R * lambda2_R);
            if (arg_R > 0) {
                S_total += n_R * (Math.log(arg_R) + 1);
            }
        }
        
        return k_B * S_total;
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
        
        // Check particle-particle collisions
        this.checkParticleCollisions();
        
        this.updateChambers();
    }
    
    step() {
        if (!this.running) return;
        this.updatePhysicsStep(0.01);
        this.updateDisplays();
        this.render();
    }
    
    updateDisplays() {
        
        const leftTemp = Math.round(this.leftChamber.calculateTemperature());
        const rightTemp = Math.round(this.rightChamber.calculateTemperature());
        const tempDiff = Math.abs(leftTemp - rightTemp);
        
        // Update displays
        document.getElementById('tempLeft').textContent = leftTemp || this.initialTemperature;
        document.getElementById('tempRight').textContent = rightTemp || this.initialTemperature;
        document.getElementById('tempDiff').textContent = Math.round(tempDiff);
        
        // Calculate actual entropy change
        const entropyDecrease = this.calculateEntropyChange();
        document.getElementById('entropyViolation').textContent = entropyDecrease.toFixed(1);
        
        // Calculate elapsed time
        const elapsed = this.running ? (Date.now() - this.startTime) / 1000 : 0;
        
        // Record data for export with circular buffer
        if (this.running && elapsed > 0) {
            // Add new data
            this.dataHistory.time.push(elapsed);
            this.dataHistory.tempLeft.push(leftTemp);
            this.dataHistory.tempRight.push(rightTemp);
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
        const currentTime = Date.now();
        const timeDiff = (currentTime - this.gateOperationTime) / 1000;
        if (timeDiff > 1) {
            const opsPerSecond = this.gateOperations / timeDiff;
            document.getElementById('gateOps').textContent = Math.round(opsPerSecond);
            this.gateOperations = 0;
            this.gateOperationTime = currentTime;
        }
        
        // Update timer and energy conservation
        if (this.running) {
            const elapsed = (Date.now() - this.startTime) / 1000;
            const minutes = Math.floor(elapsed / 60);
            const seconds = Math.floor(elapsed % 60);
            document.getElementById('timer').textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            
            // Energy conservation display
            const currentEnergy = this.calculateTotalEnergy();
            const energyRatio = currentEnergy / this.initialEnergy;
            document.getElementById('energyConservation').textContent = 
                (energyRatio * 100).toFixed(1);
            
            // Demon efficiency
            const efficiency = this.totalGateCycles > 0 
                ? (this.successfulSorts / this.totalGateCycles * 100)
                : 0;
            document.getElementById('demonEfficiency').textContent = 
                Math.round(efficiency);
            
            // Information and Landauer limit
            document.getElementById('infoBits').textContent = this.informationBits;
            const landauerEnergy = this.informationBits * Math.log(2); // kT*ln(2) per bit
            document.getElementById('landauerEnergy').textContent = landauerEnergy.toFixed(1);
            
            // Erasure cost (for realistic demon)
            document.getElementById('erasureCost').textContent = this.erasureCost.toFixed(1);
            
            // Net entropy change including information costs
            // Following the correct formula:
            // ΔS_total = ΔS_system - I×ln(2)×k_B + E×ln(2)×k_B ≥ 0
            const currentEntropy = this.calculateEntropy();
            const systemEntropyChange = currentEntropy - this.initialEntropy;
            
            // Information stored in demon's memory (negative contribution - removes entropy)
            const informationEntropy = -this.informationBits * Math.log(2) * k_B;
            
            // Information erased (positive contribution - returns entropy to environment)
            const erasureEntropy = this.erasureCost * k_B;
            
            // Total entropy must increase (Second Law)
            const totalEntropyChange = systemEntropyChange + informationEntropy + erasureEntropy;
            
            document.getElementById('netEntropy').textContent = totalEntropyChange.toFixed(3);
            
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
            
            // Fluctuation probability
            const fluctProb = this.calculateFluctuationProbability();
            document.getElementById('fluctProb').textContent = fluctProb.toExponential(2);
            
            // Max entropy decrease
            const maxEntropyPercent = this.initialEntropy > 0 
                ? (this.maxEntropyDecrease / Math.abs(this.initialEntropy)) * 100 
                : 0;
            document.getElementById('maxEntropy').textContent = maxEntropyPercent.toFixed(1);
        }
    }
    
    render() {
        // Create gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, this.width, 0);
        gradient.addColorStop(0, '#0a0e1a');
        gradient.addColorStop(0.5, '#0a0a0a');
        gradient.addColorStop(1, '#1a0e0a');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        const centerX = this.width / 2;
        
        // Draw subtle chamber indicators
        this.ctx.fillStyle = 'rgba(52, 152, 219, 0.02)';
        this.ctx.fillRect(0, 0, centerX - this.gateWidth / 2, this.height);
        this.ctx.fillStyle = 'rgba(231, 76, 60, 0.02)';
        this.ctx.fillRect(centerX + this.gateWidth / 2, 0, this.width / 2, this.height);
        
        // Draw gate with enhanced effects
        if (this.gateOpen) {
            // Open gate with glow effect
            const glowGradient = this.ctx.createLinearGradient(
                centerX - this.gateWidth / 2, 0,
                centerX + this.gateWidth / 2, 0
            );
            if (this.demonActive) {
                glowGradient.addColorStop(0, 'rgba(0, 255, 0, 0)');
                glowGradient.addColorStop(0.5, 'rgba(0, 255, 0, 0.2)');
                glowGradient.addColorStop(1, 'rgba(0, 255, 0, 0)');
            } else {
                glowGradient.addColorStop(0, 'rgba(150, 150, 150, 0)');
                glowGradient.addColorStop(0.5, 'rgba(150, 150, 150, 0.1)');
                glowGradient.addColorStop(1, 'rgba(150, 150, 150, 0)');
            }
            this.ctx.fillStyle = glowGradient;
            this.ctx.fillRect(centerX - this.gateWidth / 2, 0, this.gateWidth, this.height);
            
            // Draw dashed outline
            this.ctx.setLineDash([10, 5]);
            this.ctx.strokeStyle = this.demonActive ? 'rgba(0, 255, 0, 0.8)' : 'rgba(150, 150, 150, 0.6)';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(centerX - this.gateWidth / 2, 0, this.gateWidth, this.height);
            this.ctx.setLineDash([]);
        } else {
            // Closed gate with metallic appearance
            const metalGradient = this.ctx.createLinearGradient(
                centerX - this.gateWidth / 2, 0,
                centerX + this.gateWidth / 2, 0
            );
            metalGradient.addColorStop(0, '#333');
            metalGradient.addColorStop(0.5, '#555');
            metalGradient.addColorStop(1, '#333');
            this.ctx.fillStyle = metalGradient;
            this.ctx.fillRect(centerX - this.gateWidth / 2, 0, this.gateWidth, this.height);
            
            // Add warning stripes
            this.ctx.save();
            this.ctx.clip(new Path2D(`M${centerX - this.gateWidth / 2} 0 h${this.gateWidth} v${this.height} h-${this.gateWidth} z`));
            this.ctx.strokeStyle = 'rgba(255, 0, 0, 0.6)';
            this.ctx.lineWidth = 3;
            for (let y = -this.height; y < this.height * 2; y += 15) {
                this.ctx.beginPath();
                this.ctx.moveTo(centerX - this.gateWidth, y);
                this.ctx.lineTo(centerX + this.gateWidth, y + 30);
                this.ctx.stroke();
            }
            this.ctx.restore();
        }
        
        // Draw particles with enhanced effects
        const meanSpeed = this.calculateMeanSpeed();
        
        // First pass: draw trails and glows
        for (const particle of this.particles) {
            const speedRatio = particle.speed / meanSpeed;
            const isFast = speedRatio > 1;
            
            // Draw glow for fast particles
            if (isFast && speedRatio > 1.2) {
                const glowRadius = particle.radius * 3;
                const glowGradient = this.ctx.createRadialGradient(
                    particle.x, particle.y, 0,
                    particle.x, particle.y, glowRadius
                );
                glowGradient.addColorStop(0, `hsla(0, 100%, 60%, ${0.3 * (speedRatio - 1)})`);
                glowGradient.addColorStop(1, 'hsla(0, 100%, 60%, 0)');
                this.ctx.fillStyle = glowGradient;
                this.ctx.beginPath();
                this.ctx.arc(particle.x, particle.y, glowRadius, 0, Math.PI * 2);
                this.ctx.fill();
            }
            
            // Draw trails in slow motion
            if (this.slowMotion && particle.trail.length > 0) {
                // Get base color and convert to RGBA
                const baseColor = particle.getColor(meanSpeed);
                const speedRatio = particle.speed / meanSpeed;
                const hue = speedRatio > 1 ? 0 : 240;
                const intensity = Math.min(speedRatio > 1 ? speedRatio - 1 : 1 - speedRatio, 1);
                const lightness = 50 + intensity * 30;
                
                const gradient = this.ctx.createLinearGradient(
                    particle.trail[0].x, particle.trail[0].y,
                    particle.x, particle.y
                );
                gradient.addColorStop(0, `hsla(${hue}, 100%, ${lightness}%, 0)`);
                gradient.addColorStop(1, `hsla(${hue}, 100%, ${lightness}%, 0.4)`);
                
                this.ctx.strokeStyle = gradient;
                this.ctx.lineWidth = 2;
                this.ctx.lineCap = 'round';
                this.ctx.beginPath();
                this.ctx.moveTo(particle.trail[0].x, particle.trail[0].y);
                for (let i = 1; i < particle.trail.length; i++) {
                    this.ctx.lineTo(particle.trail[i].x, particle.trail[i].y);
                }
                this.ctx.stroke();
            }
        }
        
        // Second pass: draw particles
        for (const particle of this.particles) {
            // Main particle body
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = particle.getColor(meanSpeed);
            this.ctx.fill();
            
            // Add subtle highlight
            const highlightGradient = this.ctx.createRadialGradient(
                particle.x - particle.radius * 0.3, 
                particle.y - particle.radius * 0.3, 
                0,
                particle.x, particle.y, particle.radius
            );
            highlightGradient.addColorStop(0, 'rgba(255, 255, 255, 0.3)');
            highlightGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
            this.ctx.fillStyle = highlightGradient;
            this.ctx.fill();
        }
        
        // Draw demon at top
        this.ctx.font = 'bold 32px monospace';
        if (this.demonActive) {
            this.ctx.fillStyle = '#ffd700';
            this.ctx.fillText('👹', centerX - 16, 40);
            
            // Gate status indicator
            this.ctx.font = 'bold 14px monospace';
            this.ctx.fillStyle = this.gateOpen ? '#00ff00' : '#ff0000';
            this.ctx.fillText(this.gateOpen ? 'OPEN' : 'CLOSED', centerX - 25, 60);
            
            // Show theoretical mean speed threshold
            const theoreticalMean = this.calculateTheoreticalMeanSpeed();
            const actualMean = this.calculateMeanSpeed();
            this.ctx.font = '11px monospace';
            this.ctx.fillStyle = '#888';
            this.ctx.fillText(`Threshold: ${theoreticalMean.toFixed(0)} px/s`, centerX - 50, 80);
            this.ctx.fillText(`Actual mean: ${actualMean.toFixed(0)} px/s`, centerX - 50, 95);
        } else {
            this.ctx.fillStyle = '#666';
            this.ctx.fillText('😴', centerX - 16, 40);
            
            // Demon inactive indicator
            this.ctx.font = 'bold 14px monospace';
            this.ctx.fillStyle = '#666';
            this.ctx.fillText('INACTIVE', centerX - 30, 60);
        }
        
        // Draw velocity distribution histogram
        if (this.showHistogram) {
            this.drawVelocityHistogram();
        }
        
        // Draw phase space plot
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
    
    drawVelocityHistogram() {
        // Position and size
        const histX = HISTOGRAM_X;
        const histY = this.height - HISTOGRAM_Y_OFFSET;
        const histWidth = HISTOGRAM_WIDTH;
        const histHeight = HISTOGRAM_HEIGHT;
        
        // Background
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(histX - 10, histY - 10, histWidth + 20, histHeight + 20);
        
        // Create velocity bins
        const maxSpeed = Math.max(...this.particles.map(p => p.speed));
        this.maxHistogramSpeed = Math.max(this.maxHistogramSpeed, maxSpeed * 1.2);
        const binSize = this.maxHistogramSpeed / this.histogramBins;
        const bins = new Array(this.histogramBins).fill(0);
        
        // Count particles in each bin
        this.particles.forEach(p => {
            const binIndex = Math.min(Math.floor(p.speed / binSize), this.histogramBins - 1);
            if (binIndex >= 0) bins[binIndex]++;
        });
        
        // Find max count for scaling
        const maxCount = Math.max(...bins, 1);
        
        // Draw histogram bars
        const barWidth = histWidth / this.histogramBins;
        this.ctx.fillStyle = 'rgba(100, 150, 255, 0.7)';
        
        for (let i = 0; i < this.histogramBins; i++) {
            const barHeight = (bins[i] / maxCount) * histHeight;
            this.ctx.fillRect(
                histX + i * barWidth,
                histY + histHeight - barHeight,
                barWidth - 1,
                barHeight
            );
        }
        
        // Draw theoretical Maxwell-Boltzmann curve
        this.ctx.strokeStyle = '#ff0';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        const avgTemp = (this.leftChamber.calculateTemperature() + 
                        this.rightChamber.calculateTemperature()) / 2;
        
        for (let i = 0; i <= 100; i++) {
            const v = (i / 100) * this.maxHistogramSpeed;
            const v_norm = v / VELOCITY_SCALE;
            
            // 2D Maxwell-Boltzmann: P(v) = (m/kT) * v * exp(-mv²/2kT)
            const factor = MASS / (k_B * avgTemp / T_REF);
            const prob = factor * v_norm * Math.exp(-factor * v_norm * v_norm / 2);
            
            // Scale to histogram
            const probScaled = prob * this.particles.length * binSize / VELOCITY_SCALE;
            const y = histY + histHeight - (probScaled / maxCount) * histHeight;
            
            if (i === 0) {
                this.ctx.moveTo(histX, y);
            } else {
                this.ctx.lineTo(histX + (i / 100) * histWidth, y);
            }
        }
        this.ctx.stroke();
        
        // Labels
        this.ctx.fillStyle = '#fff';
        this.ctx.font = '12px monospace';
        this.ctx.fillText('Velocity Distribution', histX, histY - 15);
        
        // Axis labels
        this.ctx.font = '10px monospace';
        this.ctx.fillStyle = '#888';
        this.ctx.fillText('0', histX - 5, histY + histHeight + 15);
        this.ctx.fillText(`${this.maxHistogramSpeed.toFixed(0)}`, 
                          histX + histWidth - 20, histY + histHeight + 15);
        this.ctx.fillText('Speed (px/s)', histX + histWidth/2 - 30, histY + histHeight + 15);
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
        
        // Create density map for better visualization
        const gridSize = 10;
        const densityGrid = this.createDensityGrid(gridSize);
        
        // Draw density heatmap
        const maxDensity = Math.max(...densityGrid.flat());
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