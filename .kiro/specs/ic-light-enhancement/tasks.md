# Implementation Plan

- [x] 1. Set up project dependencies and environment






  - Install new Python packages (openai, basicsr, realesrgan, facexlib, gfpgan)
  - Verify Real-ESRGAN model exists at `D:\new1\light\models\RealESRGAN_x4plus.pth`
  - Set up OPENAI_API_KEY environment variable
  - _Requirements: 6.1_

- [x] 2. Implement Real-ESRGAN upscaling module




  - [x] 2.1 Create upscaler.py with ImageUpscaler class


    - Implement `_initialize_model()` method to load RealESRGAN_x4plus model
    - Implement `upscale_to_1080p()` method with proper error handling
    - Add fallback logic for GPU memory errors
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 2.2 Write unit tests for upscaler module



    - Test upscaling images below 1080p
    - Test upscaling images already above 1080p
    - Test aspect ratio preservation
    - _Requirements: 4.6_

- [x] 3. Implement GPT Vision API client for AI recommendations


  - [x] 3.1 Create gpt_recommendations.py with GPTRecommendationClient class


    - Implement `image_to_base64()` method for image encoding
    - Implement `get_lighting_recommendations()` method with GPT-4o-mini API call
    - Add error handling for API timeouts, rate limits, and network errors
    - Implement fallback to default suggestions on API failure
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

  - [x] 3.2 Write unit tests for GPT client




    - Mock API responses and test parsing
    - Test base64 encoding of various image formats
    - Test error handling scenarios
    - _Requirements: 2.5_

- [x] 4. Update gradio_demo.py with modern UI and AI recommendations






  - [x] 4.1 Add custom CSS styling

    - Create custom_css variable with dark theme colors
    - Apply gradient backgrounds and modern button styles
    - Style input fields, sliders, and image containers
    - Apply CSS to Gradio Blocks with `gr.Blocks(css=custom_css)`
    - _Requirements: 1.1, 1.2, 1.3, 1.4_


  - [x] 4.2 Add AI Recommendations UI components

    - Add "AI Recommendations" button below image upload
    - Add Radio component for displaying 3 AI suggestions (initially hidden)
    - Add "Apply" button for selected suggestion (initially hidden)
    - _Requirements: 2.1, 2.6_


  - [x] 4.3 Implement AI recommendations workflow

    - Create `get_ai_recommendations()` function that calls GPTRecommendationClient
    - Create `apply_ai_suggestion()` function to populate prompt field
    - Wire button click events to show/hide components and update prompt
    - _Requirements: 2.6, 2.7, 2.8_


  - [x] 4.4 Integrate upscaler into relighting pipeline

    - Import ImageUpscaler in gradio_demo.py
    - Initialize upscaler instance at startup
    - Modify `process_relight()` to call upscaler on all outputs before returning
    - _Requirements: 4.7, 4.8_

- [x] 5. Update gradio_demo_bg.py with background blending and modern UI









  - [x] 5.1 Add custom CSS styling (same as gradio_demo.py)

    - Apply the same custom_css variable
    - Ensure consistent styling across both demo files
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_


  - [x] 5.2 Add background image upload UI

    - Add "Add Background Image" button in foreground upload section
    - Make background image upload component initially hidden
    - Add "Blend" button (initially hidden)
    - Implement show/hide logic when "Add Background Image" is clicked
    - _Requirements: 3.1, 3.2, 3.3_


  - [x] 5.3 Add AI Recommendations to background demo

    - Add same AI recommendation components as in gradio_demo.py
    - Implement AI recommendations workflow for background mode
    - _Requirements: 2.1, 2.6, 2.7, 2.8_


  - [x] 5.4 Implement background blending workflow

    - Modify `process_relight()` to handle background image input
    - Ensure iclight_sd15_fbc.safetensors model is used for blending
    - Integrate AI recommendations as automatic prompt when blending
    - Add validation for missing background image

    - _Requirements: 3.4, 3.5, 3.6, 3.7, 3.8_


  - [ ] 5.5 Integrate upscaler into background blending pipeline
    - Import ImageUpscaler in gradio_demo_bg.py
    - Initialize upscaler instance at startup
    - Modify `process_relight()` and `process_normal()` to call upscaler on outputs
    - _Requirements: 4.7, 4.8_

- [x] 6. Implement optional interactive light direction control




  - [x] 6.1 Create JavaScript light pointer component

    - Create `light/static/light_pointer.js` file
    - Implement LightPointerControl class with draggable pointer
    - Implement coordinate-to-direction mapping logic
    - Add visual feedback for current direction
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_


  - [x] 6.2 Integrate light pointer into gradio_demo.py

    - Add custom JavaScript to load light_pointer.js
    - Add hidden Textbox component for direction value
    - Wire JavaScript callback to update hidden field
    - Modify relighting logic to use direction from hidden field
    - _Requirements: 5.8, 5.9_

- [x] 7. Update requirements.txt and documentation



  - Add new dependencies to requirements.txt (openai, basicsr, realesrgan, facexlib, gfpgan)
  - Update README.md with new features documentation
  - Add setup instructions for OPENAI_API_KEY
  - Document new UI features and workflows
  - _Requirements: 6.1_

- [x] 8. Integration testing and validation




  - [x] 8.1 Test complete workflow: upload → AI recommendations → apply → relight → verify 1080p output

    - Test with various image sizes and formats
    - Verify upscaling produces 1080p minimum height
    - Verify AI recommendations are relevant
    - _Requirements: 6.2, 6.3, 6.4_

  - [x] 8.2 Test background blending workflow

    - Upload foreground and background images
    - Get AI recommendations
    - Apply blend and verify output quality
    - Verify automatic upscaling
    - _Requirements: 6.3, 6.4_

  - [x] 8.3 Test error handling

    - Test with invalid API key
    - Test with missing Real-ESRGAN model
    - Test with network errors
    - Verify user-friendly error messages
    - _Requirements: 6.7_


  - [x] 8.4 Performance testing



    - Measure end-to-end pipeline latency
    - Verify total processing time < 30 seconds
    - Monitor GPU memory usage
    - _Requirements: 6.6_


- [x] 9. Final UI polish and cross-feature integration



  - Verify consistent styling across all pages
  - Test switching between text-conditioned and background-conditioned modes
  - Verify state consistency when using multiple features in sequence
  - Test all button interactions and component visibility
  - _Requirements: 1.4, 1.5, 6.2, 6.5_
