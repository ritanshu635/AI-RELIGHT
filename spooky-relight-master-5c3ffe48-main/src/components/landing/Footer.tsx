import { Ghost, Github, Twitter } from "lucide-react";

const Footer = () => {
  return (
    <footer className="py-12 px-4 border-t border-border bg-background">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-3">
            <Ghost className="w-8 h-8 text-primary" />
            <span className="font-creepy text-2xl text-foreground">Spooky Relight</span>
          </div>
          
          <p className="text-muted-foreground text-sm">
            Powered by IC-Light • Built for the darkness
          </p>
          
          <div className="flex gap-4">
            <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
              <Github className="w-5 h-5" />
            </a>
            <a href="#" className="text-muted-foreground hover:text-primary transition-colors">
              <Twitter className="w-5 h-5" />
            </a>
          </div>
        </div>
        
        <div className="mt-8 pt-8 border-t border-border/50 text-center text-muted-foreground text-xs">
          © {new Date().getFullYear()} Spooky Relight. All rights reserved. May your shadows be eternal.
        </div>
      </div>
    </footer>
  );
};

export default Footer;
