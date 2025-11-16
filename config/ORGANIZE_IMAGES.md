# Image Organization Plan

## Images Used by Application (Keep in root or assets/ui/)
- `logo2.png` - Logo (used in main.py)
- `external_link.png` - Icon (used in main.py)
- `resulting_sutures.png` - Example results (used in main.py)
- `example_uploads.png` - Example uploads (used in main.py)

## Suggested Organization Structure

### Option 1: Simple (Recommended)
Create these directories and move files:

```
assets/
  ui/              - UI assets (logo, icons)
  examples/        - Example images shown in UI
  test_data/       - Test/experiment images
  generated/       - Generated output images
```

### Option 2: Keep Current Structure
Just move scattered root-level images into existing directories:
- Move `shortS.png`, `longS.png`, `Vshape_far.png` → `chicken_skin_8-21-25/` or `assets/examples/`
- Keep `images/` for generated outputs
- Keep `plots/` for plot outputs
- Keep `extra_images/` for additional examples

## Files to Organize

### Root Level Images (scattered)
- `logo.png` - Old logo? (check if used)
- `logo2.png` - **KEEP** (used in UI)
- `external_link.png` - **KEEP** (used in UI)
- `resulting_sutures.png` - **KEEP** (used in UI)
- `example_uploads.png` - **KEEP** (used in UI)
- `example_uploads_v0.png` - Old version? (can move to Archive or delete)
- `shortS.png`, `shortS_far.png` - Move to examples
- `longS.png` - Move to examples
- `Vshape_far.png` - Move to examples

### Directories to Keep As-Is
- `temp_images/` - Runtime generated (auto-created)
- `plots/` - Generated plots (keep)
- `images/` - Generated outputs (255 files - consider archiving old ones)

### Directories That Are Examples/Test Data
- `chicken_images/` - Test data (can keep or move to `assets/test_data/`)
- `chicken_skin_8-21-25/` - Test data (can keep or move to `assets/test_data/`)
- `extra_images/` - Examples (can keep or move to `assets/examples/`)
- `real_images/` - Examples (can keep or move to `assets/examples/`)
- `dan/` - Test data (can keep or move to `assets/test_data/`)

## Quick Action Plan

1. **Create organization structure:**
   ```bash
   mkdir -p assets/ui assets/examples assets/test_data
   ```

2. **Move UI assets:**
   ```bash
   mv logo2.png external_link.png resulting_sutures.png example_uploads.png assets/ui/
   ```

3. **Update main.py to reference new paths:**
   - Change `Image.open("logo2.png")` → `Image.open("assets/ui/logo2.png")`
   - Change `Image.open("external_link.png")` → `Image.open("assets/ui/external_link.png")`
   - Change `Image.open("resulting_sutures.png")` → `Image.open("assets/ui/resulting_sutures.png")`
   - Change `Image.open("example_uploads.png")` → `Image.open("assets/ui/example_uploads.png")`

4. **Move scattered example images:**
   ```bash
   mv shortS.png shortS_far.png longS.png Vshape_far.png assets/examples/
   mv extra_images/ real_images/ assets/examples/  # or keep separate
   ```

5. **Optional: Archive old generated images:**
   ```bash
   mkdir -p archive/generated_images
   # Move old images/ files if needed
   ```

