import { useState, useRef, useCallback } from "react";
import { useToast } from "@/hooks/use-toast";
import EditorHeader from "@/components/editor/EditorHeader";
import ImageCompare from "@/components/editor/ImageCompare";
import ControlPanel, { ControlSettings } from "@/components/editor/ControlPanel";
import samplePortrait from "@/assets/sample-portrait.jpg";
import { Helmet } from "react-helmet-async";

const Editor = () => {
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [originalImage, setOriginalImage] = useState(samplePortrait);
  const [editedImage, setEditedImage] = useState(samplePortrait);
  const [currentSettings, setCurrentSettings] = useState<ControlSettings | null>(null);
  const [aiSuggestion, setAiSuggestion] = useState<string | null>(null);
  const [isCompareMode, setIsCompareMode] = useState(false);

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const result = event.target?.result as string;
        setOriginalImage(result);
        setEditedImage(result);
        toast({
          title: "Image uploaded!",
          description: "Your image is ready for haunting transformations.",
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleReset = () => {
    setEditedImage(originalImage);
    setAiSuggestion(null);
    toast({
      title: "Reset complete",
      description: "All edits have been undone.",
    });
  };

  const handlePresetChange = useCallback((preset: string) => {
    // Simulate applying preset (in real app, this would call the IC-Light API)
    toast({
      title: `${preset.charAt(0).toUpperCase() + preset.slice(1)} preset applied!`,
      description: "Lighting has been transformed.",
    });
    // Here you would apply the actual preset transformation
  }, [toast]);

  const handleSettingsChange = useCallback((settings: ControlSettings) => {
    setCurrentSettings(settings);
    // In real implementation, this would trigger IC-Light processing
  }, []);

  const handleAiRecommend = useCallback(async () => {
    // Simulate AI recommendation (would call GPT-4o-mini in real app)
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const suggestions = [
      "Try adding soft warm light from the top-left with subtle rim glow for a mysterious candlelit effect",
      "Apply dramatic side lighting with deep shadows for a horror movie aesthetic",
      "Use cool blue fill light with orange key light for a cinematic supernatural look",
      "Add ghostly green rim light from behind with soft frontal fill for an ethereal appearance",
    ];
    
    const randomSuggestion = suggestions[Math.floor(Math.random() * suggestions.length)];
    setAiSuggestion(randomSuggestion);
    
    toast({
      title: "AI Recommendation Ready!",
      description: randomSuggestion,
    });
  }, [toast]);

  const handleExport = useCallback(() => {
    toast({
      title: "Exporting...",
      description: "Your haunted masterpiece will be ready shortly.",
    });
    // In real implementation, this would trigger export
  }, [toast]);

  return (
    <>
      <Helmet>
        <title>Editor - Spooky Relight</title>
        <meta name="description" content="Edit your photos with AI-powered relighting controls" />
      </Helmet>
      
      <div className="h-screen flex flex-col bg-background overflow-hidden">
        <EditorHeader onUpload={handleUpload} onReset={handleReset} />
        
        <div className="flex-1 flex overflow-hidden">
          {/* Main Editor Area */}
          <div className="flex-1 p-6 flex flex-col gap-4 overflow-hidden">
            {/* AI Suggestion Banner */}
            {aiSuggestion && (
              <div className="bg-primary/10 border border-primary/30 rounded-lg p-4 animate-fade-in">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <span className="text-primary text-lg">✨</span>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-creepy text-primary text-sm mb-1">AI Suggestion</h4>
                    <p className="text-foreground text-sm">{aiSuggestion}</p>
                  </div>
                  <button
                    onClick={() => setAiSuggestion(null)}
                    className="text-muted-foreground hover:text-foreground transition-colors"
                  >
                    ×
                  </button>
                </div>
              </div>
            )}

            {/* Image View */}
            <div className="flex-1 flex items-center justify-center">
              {isCompareMode ? (
                <ImageCompare
                  originalImage={originalImage}
                  editedImage={editedImage}
                  className="max-w-4xl w-full h-full max-h-[600px]"
                />
              ) : (
                <div className="relative max-w-4xl w-full h-full max-h-[600px] rounded-xl overflow-hidden border-2 border-border bg-card">
                  <img
                    src={editedImage}
                    alt="Edited"
                    className="w-full h-full object-contain"
                  />
                </div>
              )}
            </div>

            {/* Quick Info & Compare Toggle */}
            <div className="flex justify-center items-center gap-6 text-sm text-muted-foreground">
              <button
                onClick={() => setIsCompareMode(!isCompareMode)}
                className={`px-4 py-2 rounded-lg border transition-all duration-300 ${
                  isCompareMode
                    ? "border-primary bg-primary/10 text-primary shadow-glow"
                    : "border-border hover:border-muted-foreground"
                }`}
              >
                {isCompareMode ? "Exit Compare" : "Compare Original"}
              </button>
              {isCompareMode && <span>Drag slider to compare</span>}
            </div>
          </div>

          {/* Control Panel */}
          <aside className="w-80 border-l border-border bg-card/30 p-6 overflow-hidden flex flex-col">
            <ControlPanel
              onPresetChange={handlePresetChange}
              onSettingsChange={handleSettingsChange}
              onAiRecommend={handleAiRecommend}
              onExport={handleExport}
            />
          </aside>
        </div>

        {/* Hidden file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
    </>
  );
};

export default Editor;
