# Reinforced Concrete Continuous Floor Beam Design Moment Calculator

For an Engineering Math Assignment

This Python program solves reinforced-concrete continuous floor beam design-moment problems using ACI moment coefficients and ASCE 7 live-load reduction.

## Features

- **ACI Moment Coefficient Method**: Validates applicability and computes design moments
- **ASCE 7 Live Load Reduction**: Configurable live load reduction based on tributary area
- **Multiple Beam Lines**: Processes multiple continuous beams in a single run
- **Face of Column Moments**: Computes moments at column faces using support shears
- **Flexible Input**: JSON-based configuration with sensible defaults

## Requirements

- Python 3.6 or higher
- Standard library only (no external dependencies)

## Usage

### Basic Usage

```bash
python3 beam_design.py input_example.json
```

### Input File Format

The input file is a JSON file with the following structure:

```json
{
  "slab_thickness_in": 6.0,
  "concrete_density_pcf": 150.0,
  "sdl_psf": 20.0,
  "live_load_psf": 60.0,
  "beam_web_width_in": 12.0,
  "beam_depth_in": 24.0,
  "column_width_in": 18.0,
  "dead_factor": 1.2,
  "live_factor": 1.6,
  "use_effective_span_ln": true,
  "effective_depth_d_in": 21.0,
  "assume_uniform_loads": true,
  "assume_prismatic_member": true,
  "span_ratio_limit": 1.20,
  "use_live_load_reduction": true,
  "aci_coefficients": {
    "end_span": {
      "neg_exterior_support_den": 24,
      "neg_first_interior_support_den": 9,
      "pos_midspan_den": 14
    },
    "interior_span": {
      "pos_midspan_den": 16,
      "neg_support_den_optional": 11
    }
  },
  "live_load_reduction": {
    "span_1": {
      "KLL": 2.0
    },
    "span_2": {
      "KLL": 2.0
    },
    "span_3": {
      "KLL": 2.0
    }
  },
  "beam_A_B_C_D": {
    "span_lengths_cc_ft": [28.0, 24.0, 28.0],
    "tributary_widths_ft": [10.0, 10.0, 10.0]
  },
  "beam_E_F_G_H": {
    "span_lengths_cc_ft": [26.0, 24.0, 26.0],
    "tributary_widths_ft": [12.0, 12.0, 12.0]
  }
}
```

### Input Parameters

#### Geometry
- `slab_thickness_in`: Slab thickness in inches (default: 6.0)
- `beam_web_width_in`: Beam web width in inches (default: 12.0)
- `beam_depth_in`: Beam total depth in inches (default: 24.0)
- `column_width_in`: Column width in inches (default: 18.0)
- `concrete_density_pcf`: Concrete density in pcf (default: 150.0)

#### Loads
- `live_load_psf`: Unreduced live load in psf (default: 60.0)
- `sdl_psf`: Superimposed dead load in psf (default: 20.0)
- `dead_factor`: Dead load factor (default: 1.2)
- `live_factor`: Live load factor (default: 1.6)

#### Beam Definitions
Each beam line requires:
- `span_lengths_cc_ft`: List of center-to-center span lengths in feet
- `tributary_widths_ft`: List of tributary widths in feet (one per span)

#### ACI Coefficients
- `aci_coefficients`: Dictionary with denominators for moment coefficients
  - `end_span`: Coefficients for end spans
  - `interior_span`: Coefficients for interior spans

#### Live Load Reduction
- `use_live_load_reduction`: Boolean to enable/disable reduction
- `live_load_reduction`: Dictionary with parameters per span
  - `KLL`: Live load element factor
  - `At_override_ft2`: Optional override for tributary area
  - `min_reduction_fraction`: Optional minimum reduction limit
  - `max_reduction_fraction`: Optional maximum reduction limit

#### Method Options
- `use_effective_span_ln`: Use ACI effective span calculation (default: true)
- `effective_depth_d_in`: Effective depth in inches (optional, defaults to 0.9 * beam_depth)
- `d_factor`: Factor for estimating d from beam depth (default: 0.9)
- `assume_uniform_loads`: Assume uniform loads (default: true)
- `assume_prismatic_member`: Assume prismatic member (default: true)
- `span_ratio_limit`: Maximum ratio of adjacent spans (default: 1.20)

## Output

The program outputs:

1. **ACI Coefficient Applicability Check**: YES/NO with detailed reasons
2. **Load Summary**: For each span:
   - Center-to-center and effective span lengths
   - Dead and live line loads
   - Factored uniform load
   - Live load reduction details (if enabled)
3. **Design Moments**: Factored design moments at critical sections:
   - End span: exterior negative, positive midspan, first interior negative
   - Interior span: positive midspan
   - Interior supports: negative moment at centerline and face of column

All moments are reported in **kip-ft** and shears in **kips**.

## Engineering Methods

### ACI Moment Coefficients
The program uses the standard ACI moment coefficient method:
- M = (wu × ln²) / denominator
- Default denominators follow common ACI practice but are fully configurable

### Live Load Reduction
Uses ASCE 7-style reduction:
- L = L₀ × (0.25 + 15 / √(KLL × At))
- Capped at L₀ (cannot exceed original)
- Optional minimum and maximum limits

### Effective Span Length
- ln = min(L_cc, L_clear + d)
- Where L_clear = L_cc - column_width
- d can be provided or estimated as d_factor × beam_depth

### Face of Column Moment
- Computes support moment from governing adjacent span
- Calculates support shear from end moments and uniform load
- Shifts moment to face: M_face = M_support - V_support × (col_width/2)

## Example

See `input_example.json` for a complete example configuration.

Run with:
```bash
python3 beam_design.py input_example.json
```

## Notes

- The program validates ACI coefficient applicability but will still compute moments even if checks fail (with a warning)
- Live load reduction is computed per span and can vary based on tributary area
- Support moments are taken as the larger magnitude from adjacent spans
- All calculations follow standard ACI and ASCE 7 procedures
