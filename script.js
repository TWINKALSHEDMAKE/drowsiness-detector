// Global variables
let model;
let video;
let canvas;
let ctx;
let isRunning = false;
let animationId;

// Drowsiness detection variables
let frameCount = 0;
let drowsyEvents = 0;
let alertSent = false;
const EAR_THRESHOLD = 0.25;
const CONSEC_FRAMES = 20;

// Performance tracking
let frameTimes = [];

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const testSoundBtn = document.getElementById('testSoundBtn');
const alertStatus = document.getElementById('alertStatus');
const earValue = document.getElementById('earValue');
const frameCountElement = document.getElementById('frameCount');
const drowsyEventsElement = document.getElementById('drowsyEvents');
const detectionRateElement = document.getElementById('detectionRate');
const alertList = document.getElementById('alertList');

// Initialize the application
async function init() {
    try {
        // Load the face landmarks model
        console.log('Loading face detection model...');
        model = await faceLandmarksDetection.load(
            faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
        );
        console.log('Model loaded successfully');
        
        // Setup video stream
        video = document.getElementById('video');
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = stream;
        
        // Set canvas dimensions to match video
        video.addEventListener('loadedmetadata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        });
        
        // Setup event listeners
        startBtn.addEventListener('click', startDetection);
        stopBtn.addEventListener('click', stopDetection);
        testSoundBtn.addEventListener('click', testAlertSound);
        
    } catch (error) {
        console.error('Error initializing application:', error);
        alert('Error initializing application: ' + error.message);
    }
}

// Start drowsiness detection
function startDetection() {
    if (isRunning) return;
    
    isRunning = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    
    // Reset counters
    frameCount = 0;
    drowsyEvents = 0;
    alertSent = false;
    alertList.innerHTML = '';
    updateStats();
    
    // Start detection loop
    detectDrowsiness();
    
    // Update status
    alertStatus.className = 'alert-status alert-normal';
    alertStatus.innerHTML = '<h3>Status: Normal</h3><p>Eyes are open and alert</p>';
    
    console.log('Drowsiness detection started');
}

// Stop drowsiness detection
function stopDetection() {
    if (!isRunning) return;
    
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Update status
    alertStatus.className = 'alert-status';
    alertStatus.innerHTML = '<h3>Status: Inactive</h3><p>Detection stopped</p>';
    
    console.log('Drowsiness detection stopped');
}

// Main detection loop
async function detectDrowsiness() {
    if (!isRunning) return;
    
    const startTime = performance.now();
    
    try {
        // Detect faces
        const predictions = await model.estimateFaces({
            input: video,
            returnTensors: false,
            flipHorizontal: false,
            predictIrises: true
        });
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (predictions.length > 0) {
            // Get the first face detected
            const face = predictions[0];
            
            // Draw facial landmarks (for visualization)
            drawFaceLandmarks(face);
            
            // Calculate eye aspect ratio
            const ear = calculateEAR(face);
            earValue.textContent = ear.toFixed(3);
            
            // Check for drowsiness
            if (ear < EAR_THRESHOLD) {
                frameCount++;
                
                if (frameCount >= CONSEC_FRAMES) {
                    drowsyEvents++;
                    frameCount = 0;
                    
                    // Add to alert history
                    const now = new Date();
                    const timeString = now.toLocaleTimeString();
                    const listItem = document.createElement('li');
                    listItem.textContent = `Drowsiness detected at ${timeString}`;
                    alertList.appendChild(listItem);
                    
                    // Play alert sound
                    playAlertSound();
                    
                    // Update status
                    if (drowsyEvents >= 4 && !alertSent) {
                        alertStatus.className = 'alert-status alert-danger';
                        alertStatus.innerHTML = '<h3>Status: EMERGENCY</h3><p>Multiple drowsiness events detected!</p>';
                        alertSent = true;
                        
                        // In a real application, you would send an SMS/notification here
                        console.log('EMERGENCY: Sending alert to emergency contacts');
                    } else {
                        alertStatus.className = 'alert-status alert-warning';
                        alertStatus.innerHTML = `<h3>Status: Drowsy</h3><p>Drowsiness detected (Event ${drowsyEvents})</p>`;
                    }
                    
                    console.log(`Drowsiness event #${drowsyEvents}`);
                }
            } else {
                frameCount = 0;
                
                // Update status to normal if not already
                if (!alertSent || drowsyEvents < 4) {
                    alertStatus.className = 'alert-status alert-normal';
                    alertStatus.innerHTML = '<h3>Status: Normal</h3><p>Eyes are open and alert</p>';
                }
            }
            
            // Update statistics
            updateStats();
        }
    } catch (error) {
        console.error('Error during detection:', error);
    }
    
    // Calculate FPS
    const endTime = performance.now();
    const frameTime = endTime - startTime;
    frameTimes.push(frameTime);
    
    // Keep only the last 30 frame times
    if (frameTimes.length > 30) {
        frameTimes.shift();
    }
    
    // Calculate average frame time and FPS
    const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    const fps = 1000 / avgFrameTime;
    detectionRateElement.textContent = `${fps.toFixed(1)} FPS`;
    
    // Continue detection loop
    animationId = requestAnimationFrame(detectDrowsiness);
}

// Calculate Eye Aspect Ratio (EAR)
function calculateEAR(face) {
    // Indices for left and right eye landmarks
    const leftEyeIndices = [33, 160, 158, 133, 153, 144];
    const rightEyeIndices = [362, 385, 387, 263, 373, 380];
    
    // Get landmarks for left eye
    const leftEye = leftEyeIndices.map(i => {
        const landmark = face.scaledMesh[i];
        return {x: landmark[0], y: landmark[1]};
    });
    
    // Get landmarks for right eye
    const rightEye = rightEyeIndices.map(i => {
        const landmark = face.scaledMesh[i];
        return {x: landmark[0], y: landmark[1]};
    });
    
    // Calculate EAR for left eye
    const leftEAR = eyeAspectRatio(leftEye);
    
    // Calculate EAR for right eye
    const rightEAR = eyeAspectRatio(rightEye);
    
    // Average the EAR values
    return (leftEAR + rightEAR) / 2;
}

// Calculate EAR for a single eye
function eyeAspectRatio(eye) {
    // Calculate distances between vertical eye landmarks
    const A = distance(eye[1], eye[5]);
    const B = distance(eye[2], eye[4]);
    
    // Calculate distance between horizontal eye landmarks
    const C = distance(eye[0], eye[3]);
    
    // The EAR is the ratio of the vertical distances to the horizontal distance
    return (A + B) / (2 * C);
}

// Calculate Euclidean distance between two points
function distance(point1, point2) {
    return Math.sqrt(
        Math.pow(point2.x - point1.x, 2) + 
        Math.pow(point2.y - point1.y, 2)
    );
}

// Draw facial landmarks on canvas
function drawFaceLandmarks(face) {
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
    ctx.fillStyle = 'rgba(0, 255, 0, 0.4)';
    ctx.lineWidth = 1;
    
    // Draw face mesh
    for (let i = 0; i < face.scaledMesh.length; i++) {
        const point = face.scaledMesh[i];
        ctx.beginPath();
        ctx.arc(point[0], point[1], 1, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    // Highlight eyes with a different color
    ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)';
    ctx.fillStyle = 'rgba(255, 255, 0, 0.4)';
    
    // Draw left eye landmarks (approximate indices)
    const leftEyeIndices = [33, 160, 158, 133, 153, 144];
    for (const i of leftEyeIndices) {
        const point = face.scaledMesh[i];
        ctx.beginPath();
        ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    // Draw right eye landmarks (approximate indices)
    const rightEyeIndices = [362, 385, 387, 263, 373, 380];
    for (const i of rightEyeIndices) {
        const point = face.scaledMesh[i];
        ctx.beginPath();
        ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// Play alert sound
function playAlertSound() {
    // Create audio context for generating alert sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.type = 'sine';
    oscillator.frequency.value = 800;
    gainNode.gain.value = 0.5;
    
    oscillator.start();
    
    // Create a beep pattern
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.3);
    
    setTimeout(() => {
        oscillator.stop();
    }, 300);
}

// Test alert sound
function testAlertSound() {
    playAlertSound();
    alert('Testing alert sound...');
}

// Update statistics display
function updateStats() {
    frameCountElement.textContent = frameCount;
    drowsyEventsElement.textContent = drowsyEvents;
}

// Initialize the application when the page loads
window.addEventListener('load', init);