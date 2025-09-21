// Face Authentication System JavaScript

class FaceAuthSystem {
    constructor() {
        this.currentMode = 'enroll';
        this.enrollPhotos = [];
        this.maxPhotos = 5;
        this.verifyPhoto = null;
        this.enrollStream = null;
        this.verifyStream = null;
        this.faceDetectionLoaded = false;

        this.init();
    }

    async init() {
        // Load face-api models for client-side face detection
        try {
            await this.loadFaceDetectionModels();
        } catch (e) {
            console.log('Face detection models not loaded, using basic capture');
        }

        // Initialize mode switching
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchMode(e.target.closest('.mode-btn').dataset.mode));
        });

        // Initialize enrollment controls
        document.getElementById('startEnrollCamera').addEventListener('click', () => this.startEnrollCamera());
        document.getElementById('captureEnrollPhoto').addEventListener('click', () => this.captureEnrollPhoto());
        document.getElementById('submitEnrollment').addEventListener('click', () => this.submitEnrollment());

        // Initialize verification controls
        document.getElementById('startVerifyCamera').addEventListener('click', () => this.startVerifyCamera());
        document.getElementById('captureVerifyPhoto').addEventListener('click', () => this.captureVerifyPhoto());
        document.getElementById('retakeVerifyPhoto').addEventListener('click', () => this.retakeVerifyPhoto());
        document.getElementById('submitVerification').addEventListener('click', () => this.submitVerification());

        // Input validation
        document.getElementById('enrollUserId').addEventListener('input', (e) => this.validateEnrollment());
        document.getElementById('verifyUserId').addEventListener('input', (e) => this.validateVerification());
    }

    async loadFaceDetectionModels() {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights';
        await faceapi.loadTinyFaceDetectorModel(MODEL_URL);
        this.faceDetectionLoaded = true;
    }

    switchMode(mode) {
        this.currentMode = mode;

        // Update buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        // Update sections
        document.getElementById('enrollSection').classList.toggle('active', mode === 'enroll');
        document.getElementById('verifySection').classList.toggle('active', mode === 'verify');

        // Stop any active cameras
        this.stopAllCameras();
    }

    // Enrollment Functions
    async startEnrollCamera() {
        try {
            const video = document.getElementById('enrollVideo');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            video.srcObject = stream;
            this.enrollStream = stream;

            // Show capture button, hide start button
            document.getElementById('startEnrollCamera').style.display = 'none';
            document.getElementById('captureEnrollPhoto').style.display = 'block';

            // Start face detection if available
            if (this.faceDetectionLoaded) {
                this.startFaceDetection('enrollVideo', 'enrollFaceOverlay');
            }

            this.showToast('success', 'Camera Started', 'Position your face in the frame');
        } catch (error) {
            this.showToast('error', 'Camera Error', 'Could not access camera: ' + error.message);
        }
    }

    async startFaceDetection(videoId, overlayId) {
        const video = document.getElementById(videoId);
        const overlay = document.getElementById(overlayId);

        const detectFace = async () => {
            if (video.srcObject && video.srcObject.active) {
                const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions());

                const statusElement = overlay.querySelector('.face-status span');
                if (detection) {
                    statusElement.textContent = 'Face detected - Ready to capture';
                    overlay.querySelector('.face-guide').style.borderColor = '#10b981';
                } else {
                    statusElement.textContent = 'No face detected';
                    overlay.querySelector('.face-guide').style.borderColor = 'rgba(255, 255, 255, 0.5)';
                }

                requestAnimationFrame(detectFace);
            }
        };

        detectFace();
    }

    captureEnrollPhoto() {
        const video = document.getElementById('enrollVideo');
        const canvas = document.getElementById('enrollCanvas');
        const ctx = canvas.getContext('2d');

        // Set canvas size to video size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0);

        // Convert to base64
        const photoData = canvas.toDataURL('image/jpeg', 0.95);

        // Add to photos array
        if (this.enrollPhotos.length < this.maxPhotos) {
            this.enrollPhotos.push(photoData);
            this.updateEnrollGallery();
            this.validateEnrollment();

            // Show success feedback
            this.showToast('success', 'Photo Captured', `${this.enrollPhotos.length}/${this.maxPhotos} photos captured`);

            // Flash effect
            this.flashEffect();
        } else {
            this.showToast('info', 'Maximum Photos', 'You have captured the maximum number of photos');
        }
    }

    updateEnrollGallery() {
        const grid = document.getElementById('enrollPhotosGrid');
        const count = document.getElementById('photoCount');

        grid.innerHTML = '';
        count.textContent = this.enrollPhotos.length;

        this.enrollPhotos.forEach((photo, index) => {
            const photoItem = document.createElement('div');
            photoItem.className = 'photo-item';
            photoItem.innerHTML = `
                <img src="${photo}" alt="Photo ${index + 1}">
                <button class="photo-delete" onclick="faceAuth.deleteEnrollPhoto(${index})">
                    <i class="fas fa-times"></i>
                </button>
            `;
            grid.appendChild(photoItem);
        });
    }

    deleteEnrollPhoto(index) {
        this.enrollPhotos.splice(index, 1);
        this.updateEnrollGallery();
        this.validateEnrollment();
        this.showToast('info', 'Photo Deleted', 'Photo has been removed');
    }

    validateEnrollment() {
        const userId = document.getElementById('enrollUserId').value.trim();
        const submitBtn = document.getElementById('submitEnrollment');

        const isValid = userId.length > 0 && this.enrollPhotos.length >= 1;
        submitBtn.disabled = !isValid;
    }

    async submitEnrollment() {
        const userId = document.getElementById('enrollUserId').value.trim();

        if (!userId) {
            this.showToast('error', 'Error', 'Please enter a User ID');
            return;
        }

        if (this.enrollPhotos.length === 0) {
            this.showToast('error', 'Error', 'Please capture at least one photo');
            return;
        }

        this.showLoading('Enrolling user...');

        try {
            const response = await fetch('/api/enroll', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: userId,
                    images: this.enrollPhotos
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                this.showStatus('enrollStatus', 'success', result.message || 'User enrolled successfully!');
                this.showToast('success', 'Success', 'User has been enrolled successfully');

                // Reset form
                setTimeout(() => {
                    this.resetEnrollment();
                }, 3000);
            } else {
                this.showStatus('enrollStatus', 'error', result.error || 'Enrollment failed');
                this.showToast('error', 'Enrollment Failed', result.error || 'Please try again');
            }
        } catch (error) {
            this.showStatus('enrollStatus', 'error', 'Network error: ' + error.message);
            this.showToast('error', 'Network Error', 'Could not connect to server');
        } finally {
            this.hideLoading();
        }
    }

    resetEnrollment() {
        document.getElementById('enrollUserId').value = '';
        this.enrollPhotos = [];
        this.updateEnrollGallery();
        this.validateEnrollment();
        this.hideStatus('enrollStatus');

        // Reset camera
        document.getElementById('startEnrollCamera').style.display = 'block';
        document.getElementById('captureEnrollPhoto').style.display = 'none';
        this.stopEnrollCamera();
    }

    // Verification Functions
    async startVerifyCamera() {
        try {
            const video = document.getElementById('verifyVideo');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            video.srcObject = stream;
            this.verifyStream = stream;

            // Show capture button, hide start button
            document.getElementById('startVerifyCamera').style.display = 'none';
            document.getElementById('captureVerifyPhoto').style.display = 'block';

            // Hide previous results
            document.getElementById('verifyResult').style.display = 'none';
            document.getElementById('confidenceMeter').style.display = 'none';

            // Start face detection if available
            if (this.faceDetectionLoaded) {
                this.startFaceDetection('verifyVideo', 'verifyFaceOverlay');
            }

            this.showToast('success', 'Camera Started', 'Position your face in the frame');
        } catch (error) {
            this.showToast('error', 'Camera Error', 'Could not access camera: ' + error.message);
        }
    }

    captureVerifyPhoto() {
        const video = document.getElementById('verifyVideo');
        const canvas = document.getElementById('verifyCanvas');
        const ctx = canvas.getContext('2d');

        // Set canvas size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame
        ctx.drawImage(video, 0, 0);

        // Convert to base64
        this.verifyPhoto = canvas.toDataURL('image/jpeg', 0.95);

        // Show preview
        document.getElementById('verifyPreviewImg').src = this.verifyPhoto;
        document.getElementById('verifyPhotoPreview').style.display = 'block';

        // Hide capture button
        document.getElementById('captureVerifyPhoto').style.display = 'none';

        // Stop camera
        this.stopVerifyCamera();

        this.validateVerification();
        this.showToast('success', 'Photo Captured', 'Ready for verification');
        this.flashEffect();
    }

    retakeVerifyPhoto() {
        this.verifyPhoto = null;
        document.getElementById('verifyPhotoPreview').style.display = 'none';
        this.startVerifyCamera();
        this.validateVerification();
    }

    validateVerification() {
        const userId = document.getElementById('verifyUserId').value.trim();
        const submitBtn = document.getElementById('submitVerification');

        const isValid = userId.length > 0 && this.verifyPhoto !== null;
        submitBtn.disabled = !isValid;
    }

    async submitVerification() {
        const userId = document.getElementById('verifyUserId').value.trim();

        if (!userId) {
            this.showToast('error', 'Error', 'Please enter a User ID');
            return;
        }

        if (!this.verifyPhoto) {
            this.showToast('error', 'Error', 'Please capture a photo');
            return;
        }

        this.showLoading('Verifying identity...');

        try {
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: userId,
                    image: this.verifyPhoto
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayVerificationResult(result);
            } else {
                this.showStatus('verifyStatus', 'error', result.error || 'Verification failed');
                this.showToast('error', 'Verification Failed', result.error || 'Please try again');
            }
        } catch (error) {
            this.showStatus('verifyStatus', 'error', 'Network error: ' + error.message);
            this.showToast('error', 'Network Error', 'Could not connect to server');
        } finally {
            this.hideLoading();
        }
    }

    displayVerificationResult(result) {
        const resultDiv = document.getElementById('verifyResult');
        const confidence = result.confidence || 0;
        const success = result.success;

        // Show confidence meter
        this.showConfidenceMeter(confidence);

        // Show result
        resultDiv.className = `verify-result ${success ? 'success' : 'failed'}`;
        resultDiv.innerHTML = `
            <div class="result-icon ${success ? 'success' : 'failed'}">
                <i class="fas fa-${success ? 'check-circle' : 'times-circle'}"></i>
            </div>
            <h3 class="result-title">${success ? 'Identity Verified' : 'Verification Failed'}</h3>
            <p class="result-message">${result.message || (success ? 'Your identity has been successfully verified' : 'Could not verify your identity')}</p>
            <div class="result-confidence">${(confidence * 100).toFixed(1)}%</div>
            <p class="result-confidence-label">Match Confidence</p>
        `;
        resultDiv.style.display = 'block';

        // Show toast
        if (success) {
            this.showToast('success', 'Verified', `Identity confirmed with ${(confidence * 100).toFixed(1)}% confidence`);
        } else {
            this.showToast('error', 'Not Verified', 'Identity could not be confirmed');
        }
    }

    showConfidenceMeter(confidence) {
        const meter = document.getElementById('confidenceMeter');
        const fill = document.getElementById('meterFill');
        const value = document.getElementById('meterValue');

        meter.style.display = 'block';

        // Animate the fill
        setTimeout(() => {
            fill.style.width = `${confidence * 100}%`;
            value.textContent = `${(confidence * 100).toFixed(1)}%`;
        }, 100);
    }

    // Camera Management
    stopEnrollCamera() {
        if (this.enrollStream) {
            this.enrollStream.getTracks().forEach(track => track.stop());
            this.enrollStream = null;
            document.getElementById('enrollVideo').srcObject = null;
        }
    }

    stopVerifyCamera() {
        if (this.verifyStream) {
            this.verifyStream.getTracks().forEach(track => track.stop());
            this.verifyStream = null;
            document.getElementById('verifyVideo').srcObject = null;
        }
    }

    stopAllCameras() {
        this.stopEnrollCamera();
        this.stopVerifyCamera();
    }

    // UI Helper Functions
    showStatus(elementId, type, message) {
        const element = document.getElementById(elementId);
        element.className = `status-message show ${type}`;
        element.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            ${message}
        `;
    }

    hideStatus(elementId) {
        const element = document.getElementById(elementId);
        element.className = 'status-message';
        element.innerHTML = '';
    }

    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const text = document.getElementById('loadingText');
        text.textContent = message;
        overlay.classList.add('show');
    }

    hideLoading() {
        document.getElementById('loadingOverlay').classList.remove('show');
    }

    showToast(type, title, message) {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <div class="toast-message">
                <div class="toast-title">${title}</div>
                <div class="toast-text">${message}</div>
            </div>
        `;

        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    flashEffect() {
        const flash = document.createElement('div');
        flash.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: white;
            z-index: 9999;
            pointer-events: none;
            animation: flash 0.3s ease;
        `;

        document.body.appendChild(flash);
        setTimeout(() => flash.remove(), 300);
    }
}

// Add flash animation to global styles
const style = document.createElement('style');
style.textContent = `
    @keyframes flash {
        0% { opacity: 0; }
        50% { opacity: 0.8; }
        100% { opacity: 0; }
    }
    @keyframes slideOut {
        to { transform: translateX(120%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Initialize the system
const faceAuth = new FaceAuthSystem();