# Requirements Document

## Introduction

This document outlines the requirements for enhancing the IC-Light image relighting application. The enhancements include a modernized user interface, AI-powered lighting recommendations via GPT-4o-mini, background image blending capabilities, automatic image upscaling to 1080p using Real-ESRGAN, and an optional interactive light direction control system. The goal is to create a more intuitive, powerful, and visually appealing image relighting tool while maintaining the core IC-Light functionality.

## Glossary

- **IC-Light System**: The existing image relighting application that manipulates illumination in images using diffusion models
- **Foreground Image**: The primary subject image that will be relit
- **Background Image**: A secondary image whose lighting characteristics will influence the foreground relighting
- **GPT Vision API**: OpenAI's GPT-4o-mini model with vision capabilities for analyzing images
- **Real-ESRGAN**: A pretrained super-resolution model for upscaling images
- **Lighting Direction**: The spatial origin of light in the scene (Left, Right, Top, Bottom, etc.)
- **Base64 Encoding**: A binary-to-text encoding scheme for transmitting image data
- **Gradio Interface**: The web-based UI framework currently used by IC-Light
- **Relighting Model**: The IC-Light diffusion model that applies lighting transformations
- **Background Blend Mode**: The IC-Light mode that uses both foreground and background images for relighting

## Requirements

### Requirement 1: Modern UI Design

**User Story:** As a user, I want a visually appealing and modern interface, so that the application is more enjoyable and professional to use.

#### Acceptance Criteria

1. WHEN THE IC-Light System loads, THE Gradio Interface SHALL display a custom color scheme with dark theme aesthetics
2. WHEN THE IC-Light System renders text elements, THE Gradio Interface SHALL apply updated typography with improved readability for all headings and labels
3. WHEN THE IC-Light System displays interactive components, THE Gradio Interface SHALL apply consistent styling to buttons, sliders, and input fields
4. THE Gradio Interface SHALL maintain visual hierarchy through color contrast and spacing throughout all pages
5. THE Gradio Interface SHALL preserve all existing functionality while applying the new visual design

### Requirement 2: AI-Powered Lighting Recommendations

**User Story:** As a user, I want AI-generated lighting suggestions for my images, so that I can quickly apply professional lighting without manual experimentation.

#### Acceptance Criteria

1. WHEN THE user uploads a Foreground Image, THE IC-Light System SHALL display an "AI Recommendations" button
2. WHEN THE user clicks the "AI Recommendations" button, THE IC-Light System SHALL encode the Foreground Image to Base64 Encoding format
3. WHEN THE Foreground Image is encoded, THE IC-Light System SHALL transmit the Base64 Encoding to the GPT Vision API
4. WHEN THE GPT Vision API receives the image, THE GPT Vision API SHALL analyze the image and return exactly three lighting suggestions
5. WHEN THE GPT Vision API generates suggestions, THE GPT Vision API SHALL limit each suggestion to a maximum of three words
6. WHEN THE IC-Light System receives AI suggestions, THE Gradio Interface SHALL display the three suggestions as selectable options
7. WHEN THE user clicks an "Apply" button for a suggestion, THE IC-Light System SHALL populate the prompt field with the selected suggestion
8. WHEN THE prompt field is populated with AI suggestion, THE IC-Light System SHALL maintain the suggestion text for subsequent relighting operations

### Requirement 3: Background Image Upload and Blending

**User Story:** As a user, I want to upload a custom background image and blend it with my foreground, so that I can create composite images with matched lighting.

#### Acceptance Criteria

1. WHEN THE IC-Light System displays the foreground upload area, THE Gradio Interface SHALL display an additional "Add Background Image" button
2. WHEN THE user clicks the "Add Background Image" button, THE Gradio Interface SHALL reveal a Background Image upload component
3. WHEN THE user uploads both Foreground Image and Background Image, THE Gradio Interface SHALL display a "Blend" button
4. WHEN THE user clicks the "Blend" button, THE IC-Light System SHALL extract the foreground subject using background removal
5. WHEN THE foreground is extracted, THE IC-Light System SHALL load the Background Blend Mode model (iclight_sd15_fbc.safetensors)
6. WHEN THE Background Blend Mode model is loaded, THE Relighting Model SHALL process both Foreground Image and Background Image together
7. WHEN THE user has active AI recommendations, THE IC-Light System SHALL automatically apply the selected AI recommendation as the prompt during blending
8. WHEN THE blending process completes, THE IC-Light System SHALL output the composited image with matched lighting

### Requirement 4: Automatic 1080p Upscaling

**User Story:** As a user, I want all output images automatically upscaled to high resolution, so that I receive professional-quality results without additional steps.

#### Acceptance Criteria

1. WHEN THE Relighting Model generates an output image, THE IC-Light System SHALL pass the output to the Real-ESRGAN upscaling pipeline
2. WHEN THE Real-ESRGAN receives an image, THE Real-ESRGAN SHALL load the RealESRGAN_x4plus pretrained model
3. WHEN THE pretrained model is loaded, THE Real-ESRGAN SHALL apply 4x super-resolution to the image
4. WHEN THE image is upscaled, THE Real-ESRGAN SHALL verify the output height is at least 1080 pixels
5. IF THE upscaled image height is less than 1080 pixels, THEN THE Real-ESRGAN SHALL apply additional scaling to reach 1080 pixels minimum height
6. WHEN THE upscaling is complete, THE IC-Light System SHALL maintain the aspect ratio of the original image
7. WHEN THE final upscaled image is ready, THE Gradio Interface SHALL display the 1080p image to the user
8. THE IC-Light System SHALL apply upscaling to all output images regardless of the relighting mode used

### Requirement 5: Interactive Light Direction Control (Optional)

**User Story:** As a user, I want to visually position the light source by dragging a pointer on the image, so that I can intuitively control lighting direction without selecting predefined options.

#### Acceptance Criteria

1. WHERE THE interactive light control feature is enabled, THE Gradio Interface SHALL display a draggable light pointer overlay on the Foreground Image preview
2. WHERE THE interactive light control feature is enabled, WHEN THE user drags the light pointer, THE Gradio Interface SHALL track the pointer coordinates in real-time
3. WHERE THE interactive light control feature is enabled, WHEN THE pointer is positioned in the left region of the image, THE IC-Light System SHALL interpret the Lighting Direction as "Left"
4. WHERE THE interactive light control feature is enabled, WHEN THE pointer is positioned in the right region of the image, THE IC-Light System SHALL interpret the Lighting Direction as "Right"
5. WHERE THE interactive light control feature is enabled, WHEN THE pointer is positioned in the top region of the image, THE IC-Light System SHALL interpret the Lighting Direction as "Top"
6. WHERE THE interactive light control feature is enabled, WHEN THE pointer is positioned in the bottom region of the image, THE IC-Light System SHALL interpret the Lighting Direction as "Bottom"
7. WHERE THE interactive light control feature is enabled, WHEN THE pointer is positioned in corner regions, THE IC-Light System SHALL interpret compound directions such as "Top Left" or "Bottom Right"
8. WHERE THE interactive light control feature is enabled, WHEN THE Lighting Direction is determined, THE IC-Light System SHALL automatically update the lighting preference parameter
9. WHERE THE interactive light control feature is enabled, THE Gradio Interface SHALL provide visual feedback showing the current interpreted Lighting Direction

### Requirement 6: System Integration and Performance

**User Story:** As a user, I want all new features to work seamlessly together, so that I have a cohesive and efficient workflow.

#### Acceptance Criteria

1. WHEN THE IC-Light System initializes, THE IC-Light System SHALL load all required models (IC-Light, Real-ESRGAN, RMBG) into GPU memory
2. WHEN THE user activates multiple features in sequence, THE IC-Light System SHALL maintain state consistency across all operations
3. WHEN THE AI recommendations are applied with background blending, THE IC-Light System SHALL use the AI-generated prompt for the Background Blend Mode
4. WHEN THE background blending produces output, THE IC-Light System SHALL automatically apply Real-ESRGAN upscaling before display
5. WHEN THE user switches between text-conditioned and background-conditioned modes, THE Gradio Interface SHALL preserve the modern UI styling
6. THE IC-Light System SHALL process images with a maximum latency of 30 seconds for the complete pipeline (relighting plus upscaling)
7. WHEN THE IC-Light System encounters errors in any feature, THE Gradio Interface SHALL display user-friendly error messages without crashing
