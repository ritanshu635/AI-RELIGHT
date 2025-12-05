import { useState, useRef, useCallback, useEffect } from "react";
import { Sun } from "lucide-react";
import { cn } from "@/lib/utils";

interface LightGizmoProps {
  onPositionChange: (position: { x: number; y: number; angle: number }) => void;
  className?: string;
}

const LightGizmo = ({ onPositionChange, className }: LightGizmoProps) => {
  const [position, setPosition] = useState({ x: 50, y: 30 });
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const calculateAngle = useCallback((x: number, y: number) => {
    const centerX = 50;
    const centerY = 50;
    const angle = Math.atan2(y - centerY, x - centerX) * (180 / Math.PI);
    return angle;
  }, []);

  const handleMove = useCallback((clientX: number, clientY: number) => {
    if (!containerRef.current || !isDragging.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(5, Math.min(95, ((clientX - rect.left) / rect.width) * 100));
    const y = Math.max(5, Math.min(95, ((clientY - rect.top) / rect.height) * 100));
    
    setPosition({ x, y });
    onPositionChange({ x, y, angle: calculateAngle(x, y) });
  }, [onPositionChange, calculateAngle]);

  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    handleMove(e.clientX, e.clientY);
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging.current) {
      handleMove(e.clientX, e.clientY);
    }
  };

  useEffect(() => {
    const handleGlobalMouseUp = () => {
      isDragging.current = false;
    };
    window.addEventListener("mouseup", handleGlobalMouseUp);
    return () => window.removeEventListener("mouseup", handleGlobalMouseUp);
  }, []);

  // Calculate light direction for the beam
  const beamEndX = 50 + (position.x - 50) * 0.3;
  const beamEndY = 50 + (position.y - 50) * 0.3;

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative w-full aspect-square rounded-xl border-2 border-border bg-card/50 overflow-hidden cursor-crosshair",
        className
      )}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseUp}
    >
      {/* Grid lines */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20">
        <line x1="50%" y1="0" x2="50%" y2="100%" stroke="currentColor" strokeWidth="1" />
        <line x1="0" y1="50%" x2="100%" y2="50%" stroke="currentColor" strokeWidth="1" />
        <circle cx="50%" cy="50%" r="30%" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="4 4" />
      </svg>

      {/* Light beam */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        <defs>
          <linearGradient id="beamGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.8" />
            <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0" />
          </linearGradient>
        </defs>
        <line
          x1={`${position.x}%`}
          y1={`${position.y}%`}
          x2={`${beamEndX}%`}
          y2={`${beamEndY}%`}
          stroke="url(#beamGradient)"
          strokeWidth="20"
          strokeLinecap="round"
          opacity="0.5"
        />
      </svg>

      {/* Center indicator */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-muted-foreground/30" />

      {/* Light source */}
      <div
        className="absolute w-12 h-12 -translate-x-1/2 -translate-y-1/2 cursor-grab active:cursor-grabbing transition-transform duration-75"
        style={{
          left: `${position.x}%`,
          top: `${position.y}%`,
        }}
      >
        <div className="relative w-full h-full">
          {/* Glow effect */}
          <div className="absolute inset-0 rounded-full bg-primary/30 animate-pulse blur-xl" />
          <div className="absolute inset-0 rounded-full bg-primary/50 blur-md" />
          
          {/* Sun icon */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center shadow-glow">
              <Sun className="w-6 h-6 text-primary-foreground" />
            </div>
          </div>
        </div>
      </div>

      {/* Direction label */}
      <div className="absolute bottom-2 left-2 right-2 text-center text-xs text-muted-foreground">
        Drag to move light source
      </div>
    </div>
  );
};

export default LightGizmo;
