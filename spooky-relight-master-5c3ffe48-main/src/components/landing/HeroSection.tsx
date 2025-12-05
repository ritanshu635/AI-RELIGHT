import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Ghost, Sparkles, Moon } from "lucide-react";
import hauntedMansion from "@/assets/haunted-mansion.jpg";

const HeroSection = () => {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Image */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${hauntedMansion})` }}
      />
      
      {/* Dark overlay with vignette */}
      <div className="absolute inset-0 bg-gradient-to-b from-background/10 via-background/30 to-background" />
      <div className="absolute inset-0 bg-spooky-vignette opacity-40" />
      
      {/* Fog effect */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
      
      {/* Floating particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-primary/30 rounded-full animate-float"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 5}s`,
              animationDuration: `${3 + Math.random() * 4}s`,
            }}
          />
        ))}
      </div>

      {/* Content */}
      <div className="relative z-10 text-center px-4 max-w-5xl mx-auto eerie-entrance">
        {/* Floating ghost icon */}
        <div className="flex justify-center mb-6">
          <Ghost className="w-16 h-16 text-glow-green ghost-float text-glow-green" />
        </div>

        {/* Main title */}
        <h1 className="font-creepy text-7xl md:text-9xl text-foreground spooky-glow mb-4 tracking-wider">
          Spooky Relight
        </h1>
        
        {/* Tagline */}
        <p className="text-xl md:text-2xl text-muted-foreground mb-2 font-serif italic">
          "Where shadows come alive..."
        </p>
        
        <p className="text-lg md:text-xl text-foreground/80 mb-12 max-w-2xl mx-auto">
          Transform your photos with AI-powered relighting. 
          Drag the light, craft the shadows, unleash the darkness within.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Button 
            variant="spooky" 
            size="xl"
            onClick={() => window.location.href = "http://localhost:5000"}
            className="group"
          >
            <Sparkles className="w-5 h-5 group-hover:animate-spin" />
            Get Started
          </Button>
          
          <Button 
            variant="ghost_btn" 
            size="lg"
            onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
          >
            <Moon className="w-4 h-4" />
            Explore Features
          </Button>
        </div>

        {/* Stats */}
        <div className="mt-16 grid grid-cols-3 gap-8 max-w-lg mx-auto">
          {[
            { label: "Light Presets", value: "13+" },
            { label: "AI Powered", value: "100%" },
            { label: "Terrifying", value: "∞" },
          ].map((stat, i) => (
            <div key={i} className="text-center">
              <div className="font-creepy text-3xl text-primary spooky-glow">{stat.value}</div>
              <div className="text-sm text-muted-foreground">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Blood drips from top */}
      <div className="absolute top-0 left-0 right-0 flex justify-around pointer-events-none">
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            className="w-2 bg-accent/60 rounded-b-full"
            style={{
              height: `${30 + Math.random() * 70}px`,
              marginLeft: `${Math.random() * 20}px`,
            }}
          />
        ))}
      </div>
    </section>
  );
};

export default HeroSection;
