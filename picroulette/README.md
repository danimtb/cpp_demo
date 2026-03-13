# PicRoulette

Real-time drawing recognition app using AI to identify hand-drawn sketches.

## Usage

1. **Start**: Run the app (shows camera feed + capture zone)
2. **Draw**: Sketch in the left capture zone (rectangle on screen)
3. **Wait**: App detects when drawing is stable (5 frames)
4. **Recognize**: AI identifies the drawing and shows "Detected: [object]"
5. **Game**: Try to draw the target object shown in green text
6. **Controls**: 
   - `R` = New random target
   - `ESC/Q` = Quit

## Configuration

Edit `assets/config.json`:

- `debug`: Enable debug windows
- `captureZone`: Drawing detection area
- `recognitionThreshold`: Min confidence (0.0-1.0)

This project uses pre-trained ONNX models fine-tuned on the [Google QuickDraw dataset](https://quickdraw.withgoogle.com/data).

## Pre-trained Models

You can used a pre-trained ONNX model fine-tuned on the [Google QuickDraw dataset](https://quickdraw.withgoogle.com/data).

- [`quickdraw-MobileNetV2-1.0-finetune`](https://huggingface.co/JoshuaKelleyDs/quickdraw-MobileNetV2-1.0-finetune)  
- [`quickdraw-MobileVITV2-2.0-Finetune`](https://huggingface.co/JoshuaKelleyDs/quickdraw-MobileVITV2-2.0-Finetune)

After downloading, place the `.onnx` files into your local `assets/models/` folder (or update your configuration file accordingly).
