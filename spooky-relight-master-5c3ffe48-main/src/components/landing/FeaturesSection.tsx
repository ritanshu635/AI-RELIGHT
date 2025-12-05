import { 
  Sun, 
  Palette, 
  Wand2, 
  SlidersHorizontal, 
  Layers, 
  Sparkles,
  Eye,
  Lightbulb
} from "lucide-react";

const features = [
  {
    icon: Sun,
    title: "Draggable Light Source",
    description: "Move the light anywhere. Watch shadows dance in real-time.",
  },
  {
    icon: Wand2,
    title: "AI Recommendations",
    description: "GPT-powered suggestions for the perfect haunting atmosphere.",
  },
  {
    icon: Palette,
    title: "Cinematic Presets",
    description: "Hollywood, Cyberpunk, Horror, and more one-click styles.",
  },
  {
    icon: SlidersHorizontal,
    title: "Shadow Control",
    description: "Fine-tune shadow intensity from subtle to sinister.",
  },
  {
    icon: Layers,
    title: "Multi-Light Setup",
    description: "Combine key, fill, and rim lights like a pro.",
  },
  {
    icon: Eye,
    title: "Side-by-Side Compare",
    description: "See your transformation instantly with split view.",
  },
  {
    icon: Lightbulb,
    title: "Material Modes",
    description: "Matte, glossy, metallic - change how light reflects.",
  },
  {
    icon: Sparkles,
    title: "Mood Generator",
    description: "Select emotions. AI crafts the perfect lighting.",
  },
];

const FeaturesSection = () => {
  return (
    <section id="features" className="py-24 px-4 bg-dark-gradient relative">
      {/* Subtle pattern overlay */}
      <div className="absolute inset-0 opacity-5 pointer-events-none" 
        style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.4"%3E%3Cpath d="M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")' }}
      />

      <div className="max-w-6xl mx-auto relative z-10">
        <div className="text-center mb-16">
          <h2 className="font-creepy text-5xl md:text-6xl text-foreground spooky-glow mb-4">
            Unholy Features
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Everything you need to transform ordinary photos into supernatural masterpieces
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group p-6 bg-card/50 backdrop-blur-sm border border-border rounded-xl hover:border-primary/50 transition-all duration-500 hover:shadow-glow hover:-translate-y-1"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors group-hover:shadow-[0_0_20px_hsl(var(--primary)/0.3)]">
                <feature.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-creepy text-xl text-foreground mb-2 group-hover:text-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-muted-foreground text-sm">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
