# Suture Planning Progress GUI

A minimalistic and clean progress bar GUI that displays real-time information during the suture placement optimization process.

## Features

- **Overall Progress Bar**: Shows the overall progress through the optimization process
- **Current Suture Optimization**: Displays which suture configuration is currently being optimized
- **Real-time Loss Values**: Shows total loss, closure loss, and shear loss for the current optimization
- **Best Result Tracking**: Keeps track of the best suture configuration found so far
- **Status Updates**: Provides clear status messages about what's happening
- **Clean Design**: Minimalistic interface with a modern look

## Components

### Progress GUI Window
The GUI window is divided into several sections:

1. **Overall Progress**: Shows the overall progress through all suture configurations
2. **Current Suture Optimization**: Shows the current suture count being tested and an animated progress bar
3. **Loss Values**: Displays the current loss components (Total, Closure, Shear)
4. **Best Result**: Shows the best suture configuration found so far
5. **Status**: Provides status messages about the current operation

## Usage

### Basic Usage
The progress GUI is automatically integrated into the main suture placement pipeline. When you run:

```bash
python main.py -i image.jpg
```

The progress GUI will automatically start and show the optimization progress.

### Testing the GUI
You can test the progress GUI independently using the test script:

```bash
python test_progress_gui.py
```

This will run a simulation of the optimization process to demonstrate the GUI functionality.

### Programmatic Usage
You can also use the progress GUI in your own code:

```python
from progress_gui import start_progress_gui, stop_progress_gui, get_progress_gui

# Start the GUI
start_progress_gui("My Custom Title")

# Get the GUI instance
progress_gui = get_progress_gui()

# Update progress
progress_gui.set_suture_range(3, 8)
progress_gui.update_current_sutures(5)
progress_gui.update_losses(2.5, 1.2, 1.3)

# Clean up
stop_progress_gui()
```

## API Reference

### SutureProgressGUI Class

#### Methods

- `start()`: Start the GUI in a separate thread
- `stop()`: Stop and close the GUI
- `update_overall_progress(progress, stage)`: Update overall progress (0.0 to 1.0)
- `set_suture_range(start_range, end_range)`: Set the range of sutures being tested
- `update_current_sutures(num_sutures)`: Update the current suture count being optimized
- `update_losses(total_loss, closure_loss, shear_loss, ...)`: Update loss values
- `set_status(status)`: Update the status message
- `mark_complete()`: Mark the optimization as complete

### Global Functions

- `start_progress_gui(title)`: Start the progress GUI with optional title
- `stop_progress_gui()`: Stop the current progress GUI
- `get_progress_gui()`: Get the current progress GUI instance

## Integration Details

The progress GUI is integrated into the `SuturePlacer.place_sutures()` method and automatically:

1. Tracks the range of suture configurations being tested
2. Updates progress as each suture configuration is optimized
3. Displays loss values for each optimization step
4. Tracks and displays the best result found
5. Shows completion status when done

## Technical Details

- **Threading**: The GUI runs in a separate thread to avoid blocking the main optimization process
- **Thread Safety**: All GUI updates are performed using `root.after()` to ensure thread safety
- **Automatic Cleanup**: The GUI is automatically cleaned up when the optimization completes
- **Error Handling**: The GUI gracefully handles cases where it's not available

## Requirements

- Python 3.6+
- tkinter (usually included with Python, but not required - console fallback available)
- threading (built-in)

## Fallback Mode

If tkinter is not available on your system, the progress GUI will automatically fall back to console output mode. In this mode, all progress information will be displayed in the terminal with clear formatting and status updates.

## Customization

You can customize the appearance and behavior by modifying the `SutureProgressGUI` class in `progress_gui.py`. The GUI uses the 'clam' theme by default, but you can change this to other available themes like 'alt', 'default', 'classic', etc. 