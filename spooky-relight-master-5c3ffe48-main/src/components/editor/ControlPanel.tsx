import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { 
  Sparkles, 
  Wand2, 
  Sun, 
  Moon, 
  Zap, 
  Skull,
  Ghost,
  Flame,
  Snowflake,
  CloudLightning,
  Clapperboard,
  Image,
  Download
} from "lucide-react";
import LightGizmo from "./LightGizmo";

interface ControlPanelProps {
  onPresetChange: (preset: string) => void;
  onSettingsChange: (settings: ControlSettings) => void;
  onAiRecommend: () => void;
  onExport: () => void;
}

export interface ControlSettings {
  brightness: number;
  shadowIntensity: number;
  warmth: number;
  contrast: number;
  lightX: number;
  lightY: number;
  lightAngle: number;
  material: string;
  prompt: string;
}

const presets = [
  { id: "horror", label: "Horror", icon: Skull, color: "text-accent" },
  { id: "ghost", label: "Ghostly", icon: Ghost, color: "text-glow-green" },
  { id: "flame", label: "Hellfire", icon: Flame, color: "text-pumpkin" },
  { id: "frost", label: "Frost", icon: Snowflake, color: "text-blue-400" },
  { id: "storm", label: "Storm", icon: CloudLightning, color: "text-glow-purple" },
  { id: "cinema", label: "Cinematic", icon: Clapperboard, color: "text-candle" },
];

const materials = [
  { id: "matte", label: "Matte" },
  { id: "glossy", label: "Glossy" },
  { id: "metallic", label: "Metallic" },
  { id: "skin", label: "Soft Skin" },
  { id: "ceramic", label: "Ceramic" },
];

const ControlPanel = ({ onPresetChange, onSettingsChange, onAiRecommend, onExport }: ControlPanelProps) => {
  const [settings, setSettings] = useState<ControlSettings>({
    brightness: 50,
    shadowIntensity: 60,
    warmth: 50,
    contrast: 50,
    lightX: 50,
    lightY: 30,
    lightAngle: 0,
    material: "matte",
    prompt: "",
  });
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const updateSetting = (key: keyof ControlSettings, value: number | string) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    onSettingsChange(newSettings);
  };

  const handlePresetClick = (presetId: string) => {
    setActivePreset(presetId);
    onPresetChange(presetId);
  };

  const handleLightPositionChange = (position: { x: number; y: number; angle: number }) => {
    const newSettings = {
      ...settings,
      lightX: position.x,
      lightY: position.y,
      lightAngle: position.angle,
    };
    setSettings(newSettings);
    onSettingsChange(newSettings);
  };

  const handleAiRecommend = async () => {
    setIsGenerating(true);
    await onAiRecommend();
    setIsGenerating(false);
  };

  return (
    <div className="h-full flex flex-col gap-6 overflow-y-auto pr-2 custom-scrollbar">
      {/* Light Position Gizmo */}
      <div className="space-y-3">
        <h3 className="font-creepy text-lg text-foreground flex items-center gap-2">
          <Sun className="w-4 h-4 text-primary" />
          Light Position
        </h3>
        <LightGizmo onPositionChange={handleLightPositionChange} />
      </div>

      {/* AI Recommendation */}
      <div className="space-y-3">
        <h3 className="font-creepy text-lg text-foreground flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-primary" />
          AI Magic
        </h3>
        <Button 
          variant="spooky" 
          className="w-full" 
          onClick={handleAiRecommend}
          disabled={isGenerating}
        >
          <Wand2 className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
          {isGenerating ? 'Conjuring...' : 'Get AI Recommendations'}
        </Button>
        <textarea
          placeholder="Or enter your own lighting prompt..."
          className="w-full h-20 px-3 py-2 bg-input border border-border rounded-lg text-foreground text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary resize-none"
          value={settings.prompt}
          onChange={(e) => updateSetting("prompt", e.target.value)}
        />
      </div>

      {/* Presets */}
      <div className="space-y-3">
        <h3 className="font-creepy text-lg text-foreground flex items-center gap-2">
          <Zap className="w-4 h-4 text-primary" />
          Haunted Presets
        </h3>
        <div className="grid grid-cols-3 gap-2">
          {presets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handlePresetClick(preset.id)}
              className={`flex flex-col items-center gap-1 p-3 rounded-lg border transition-all duration-300 ${
                activePreset === preset.id
                  ? "border-primary bg-primary/10 shadow-glow"
                  : "border-border bg-card/50 hover:border-muted-foreground"
              }`}
            >
              <preset.icon className={`w-5 h-5 ${preset.color}`} />
              <span className="text-xs text-foreground">{preset.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-4">
        <h3 className="font-creepy text-lg text-foreground flex items-center gap-2">
          <Moon className="w-4 h-4 text-primary" />
          Shadow Controls
        </h3>
        
        {[
          { key: "brightness", label: "Brightness", icon: Sun },
          { key: "shadowIntensity", label: "Shadow Depth", icon: Moon },
          { key: "warmth", label: "Warmth", icon: Flame },
          { key: "contrast", label: "Contrast", icon: Zap },
        ].map((control) => (
          <div key={control.key} className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground flex items-center gap-2">
                <control.icon className="w-3 h-3" />
                {control.label}
              </span>
              <span className="text-xs text-primary font-mono">
                {settings[control.key as keyof ControlSettings]}%
              </span>
            </div>
            <Slider
              value={[settings[control.key as keyof ControlSettings] as number]}
              onValueChange={(value) => updateSetting(control.key as keyof ControlSettings, value[0])}
              max={100}
              step={1}
            />
          </div>
        ))}
      </div>

      {/* Material Mode */}
      <div className="space-y-3">
        <h3 className="font-creepy text-lg text-foreground flex items-center gap-2">
          <Image className="w-4 h-4 text-primary" />
          Material
        </h3>
        <div className="flex flex-wrap gap-2">
          {materials.map((mat) => (
            <button
              key={mat.id}
              onClick={() => updateSetting("material", mat.id)}
              className={`px-3 py-1.5 rounded-full text-xs border transition-all ${
                settings.material === mat.id
                  ? "border-primary bg-primary/10 text-primary"
                  : "border-border text-muted-foreground hover:border-muted-foreground"
              }`}
            >
              {mat.label}
            </button>
          ))}
        </div>
      </div>

      {/* Export Button */}
      <div className="mt-auto pt-4 border-t border-border">
        <Button variant="blood" className="w-full" onClick={onExport}>
          <Download className="w-4 h-4" />
          Export Image (1080p)
        </Button>
      </div>
    </div>
  );
};

export default ControlPanel;
