// LightPointerControl - Interactive light direction control for IC-Light
class LightPointerControl {
    constructor(imageElement, callbackFn) {
        this.image = imageElement;
        this.callback = callbackFn;
        this.pointer = null;
        this.directionLabel = null;
        this.isDragging = false;
        this.currentDirection = "None";
        this.init();
    }
    
    init() {
        // Create pointer overlay
        this.pointer = document.createElement('div');
        this.pointer.className = 'light-pointer';
        this.pointer.style.cssText = `
            position: absolute;
            width: 40px;
            height: 40px;
            background: radial-gradient(circle, #FFD700 0%, #FFA500 100%);
            border-radius: 50%;
            cursor: move;
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
            z-index: 1000;
            transition: box-shadow 0.2s ease;
        `;
        
        // Create direction label
        this.directionLabel = document.createElement('div');
        this.directionLabel.className = 'light-direction-label';
        this.directionLabel.style.cssText = `
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.7);
            color: #FFD700;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            z-index: 1001;
            pointer-events: none;
        `;
        this.directionLabel.textContent = 'Direction: None';
        
        // Position pointer at center initially
        this.updatePosition(0.5, 0.5);
        
        // Add to image container
        const container = this.image.parentElement;
        container.style.position = 'relative';
        container.appendChild(this.pointer);
        container.appendChild(this.directionLabel);
        
        // Event listeners
        this.pointer.addEventListener('mousedown', this.startDrag.bind(this));
        document.addEventListener('mousemove', this.drag.bind(this));
        document.addEventListener('mouseup', this.endDrag.bind(this));
        
        // Touch events for mobile
        this.pointer.addEventListener('touchstart', this.startDrag.bind(this));
        document.addEventListener('touchmove', this.drag.bind(this));
        document.addEventListener('touchend', this.endDrag.bind(this));
    }
    
    startDrag(e) {
        this.isDragging = true;
        this.pointer.style.boxShadow = '0 0 30px rgba(255, 215, 0, 1)';
        e.preventDefault();
    }
    
    drag(e) {
        if (!this.isDragging) return;
        
        // Handle both mouse and touch events
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0].clientY);
        
        if (!clientX || !clientY) return;
        
        const rect = this.image.getBoundingClientRect();
        let x = (clientX - rect.left) / rect.width;
        let y = (clientY - rect.top) / rect.height;
        
        // Clamp to image bounds
        x = Math.max(0, Math.min(1, x));
        y = Math.max(0, Math.min(1, y));
        
        this.updatePosition(x, y);
        this.updateDirection(x, y);
        
        e.preventDefault();
    }
    
    endDrag() {
        this.isDragging = false;
        this.pointer.style.boxShadow = '0 0 20px rgba(255, 215, 0, 0.8)';
    }
    
    updatePosition(x, y) {
        const rect = this.image.getBoundingClientRect();
        this.pointer.style.left = `${x * rect.width - 20}px`;
        this.pointer.style.top = `${y * rect.height - 20}px`;
    }
    
    updateDirection(x, y) {
        // Map coordinates to direction
        let direction = "None";
        
        // Define regions with thresholds
        const leftThreshold = 0.33;
        const rightThreshold = 0.67;
        const topThreshold = 0.33;
        const bottomThreshold = 0.67;
        
        // Determine direction based on position
        if (x < leftThreshold) {
            if (y < topThreshold) {
                direction = "Top Left";
            } else if (y > bottomThreshold) {
                direction = "Bottom Left";
            } else {
                direction = "Left";
            }
        } else if (x > rightThreshold) {
            if (y < topThreshold) {
                direction = "Top Right";
            } else if (y > bottomThreshold) {
                direction = "Bottom Right";
            } else {
                direction = "Right";
            }
        } else {
            if (y < topThreshold) {
                direction = "Top";
            } else if (y > bottomThreshold) {
                direction = "Bottom";
            } else {
                direction = "None";
            }
        }
        
        // Update only if direction changed
        if (direction !== this.currentDirection) {
            this.currentDirection = direction;
            this.directionLabel.textContent = `Direction: ${direction}`;
            
            // Call callback with direction
            if (this.callback) {
                this.callback(direction);
            }
        }
    }
    
    destroy() {
        // Clean up event listeners and DOM elements
        if (this.pointer) {
            this.pointer.remove();
        }
        if (this.directionLabel) {
            this.directionLabel.remove();
        }
        document.removeEventListener('mousemove', this.drag);
        document.removeEventListener('mouseup', this.endDrag);
        document.removeEventListener('touchmove', this.drag);
        document.removeEventListener('touchend', this.endDrag);
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LightPointerControl;
}
