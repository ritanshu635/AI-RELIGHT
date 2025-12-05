import HeroSection from "@/components/landing/HeroSection";
import FeaturesSection from "@/components/landing/FeaturesSection";
import Footer from "@/components/landing/Footer";
import { Helmet } from "react-helmet-async";

const Index = () => {
  return (
    <>
      <Helmet>
        <title>Spooky Relight - AI Photo Relighting Tool</title>
        <meta name="description" content="Transform your photos with AI-powered relighting. Drag the light, craft the shadows, unleash the darkness within." />
      </Helmet>
      <main className="min-h-screen bg-background">
        <HeroSection />
        <FeaturesSection />
        <Footer />
      </main>
    </>
  );
};

export default Index;
