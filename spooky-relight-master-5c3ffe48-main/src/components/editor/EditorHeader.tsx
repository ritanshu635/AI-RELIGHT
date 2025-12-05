import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Ghost, ArrowLeft, Upload, RotateCcw } from "lucide-react";

interface EditorHeaderProps {
  onUpload: () => void;
  onReset: () => void;
}

const EditorHeader = ({ onUpload, onReset }: EditorHeaderProps) => {
  const navigate = useNavigate();

  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <Button 
          variant="ghost" 
          size="icon"
          onClick={() => navigate("/")}
        >
          <ArrowLeft className="w-5 h-5" />
        </Button>
        
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/")}>
          <Ghost className="w-6 h-6 text-primary" />
          <span className="font-creepy text-xl text-foreground">Spooky Relight</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <Button variant="outline" onClick={onReset}>
          <RotateCcw className="w-4 h-4" />
          Reset
        </Button>
        <Button variant="spooky" onClick={onUpload}>
          <Upload className="w-4 h-4" />
          Upload Image
        </Button>
      </div>
    </header>
  );
};

export default EditorHeader;
