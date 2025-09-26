// Add this at the beginning of your init() function

async function init() {
    try {
        // First, request camera access
        console.log('Requesting camera access...');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"  // Front camera
            } 
        });
        
        // Setup video element
        video = document.getElementById('video');
        canvas = document.getElementById('canvas');
        ctx = canvas.getContext('2d');
        
        video.srcObject = stream;
        
        // Wait for video to load
        await new Promise((resolve) => {
            video.addEventListener('loadedmetadata', () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            });
        });
        
        // Update camera status
        document.getElementById('cameraStatus').textContent = '🟢';
        
        // Then load the AI model
        console.log('Loading face detection model...');
        model = await faceLandmarksDetection.load(
            faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
            { maxFaces: 1 }
        );
        
        document.getElementById('modelStatus').textContent = '🟢';
        console.log('Model loaded successfully');
        
        // Rest of your existing init code...
        
    } catch (error) {
        console.error('Error initializing:', error);
        
        if (error.name === 'NotAllowedError') {
            alert('❌ Camera access denied. Please allow camera access to use this application.');
        } else {
            alert('Error initializing application: ' + error.message);
        }
    }
}