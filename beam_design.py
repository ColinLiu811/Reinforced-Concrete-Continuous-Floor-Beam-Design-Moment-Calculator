#!/usr/bin/env python3
"""
Reinforced Concrete Continuous Floor Beam Design Moment Calculator
Uses ACI moment coefficients and ASCE 7 live-load reduction.
"""

import json
import argparse
import sys
from typing import Dict, List, Tuple, Optional


def check_aci_coeff_applicability(
    num_spans: int,
    span_lengths: List[float],
    assume_uniform_loads: bool,
    assume_prismatic_member: bool,
    span_ratio_limit: float = 1.20
) -> Tuple[bool, List[str]]:
    """
    Check if ACI moment coefficients can be used.
    
    Returns:
        (is_applicable, list_of_reasons)
    """
    reasons = []
    all_passed = True
    
    # Check 1: At least 2 spans
    if num_spans >= 2:
        reasons.append("✓ Beam has ≥ 2 spans")
    else:
        reasons.append("✗ Beam must have ≥ 2 spans (has {})".format(num_spans))
        all_passed = False
    
    # Check 2: Uniform loads
    if assume_uniform_loads:
        reasons.append("✓ Loads are uniformly distributed")
    else:
        reasons.append("✗ Loads must be uniformly distributed")
        all_passed = False
    
    # Check 3: Spans approximately equal
    if len(span_lengths) >= 2:
        max_ratio = 1.0
        for i in range(len(span_lengths) - 1):
            ratio1 = max(span_lengths[i], span_lengths[i+1]) / min(span_lengths[i], span_lengths[i+1])
            max_ratio = max(max_ratio, ratio1)
        
        if max_ratio <= span_ratio_limit:
            reasons.append("✓ Adjacent span ratio ({:.2f}) ≤ {:.2f}".format(max_ratio, span_ratio_limit))
        else:
            reasons.append("✗ Adjacent span ratio ({:.2f}) > {:.2f}".format(max_ratio, span_ratio_limit))
            all_passed = False
    else:
        reasons.append("? Cannot check span equality (need ≥ 2 spans)")
    
    # Check 4: Prismatic member
    if assume_prismatic_member:
        reasons.append("✓ Member is prismatic")
    else:
        reasons.append("✗ Member must be prismatic")
        all_passed = False
    
    return all_passed, reasons


def compute_live_load_reduction(
    l0_psf: float,
    at_ft2: float,
    kll: float,
    min_reduction_fraction: Optional[float] = None,
    max_reduction_fraction: Optional[float] = None
) -> float:
    """
    Compute reduced live load using ASCE 7-style reduction.
    
    Default equation: L = L0 * (0.25 + 15 / sqrt(KLL * At))
    
    Returns:
        Reduced live load in psf
    """
    # ASCE 7 reduction equation
    reduction_factor = 0.25 + 15.0 / (kll * at_ft2) ** 0.5
    l_reduced_psf = l0_psf * reduction_factor
    
    # Cap at L0 (cannot exceed original)
    l_reduced_psf = min(l_reduced_psf, l0_psf)
    
    # Apply minimum limit if specified
    if min_reduction_fraction is not None:
        min_l_psf = l0_psf * min_reduction_fraction
        l_reduced_psf = max(l_reduced_psf, min_l_psf)
    
    # Apply maximum limit if specified
    if max_reduction_fraction is not None:
        max_l_psf = l0_psf * max_reduction_fraction
        l_reduced_psf = min(l_reduced_psf, max_l_psf)
    
    return l_reduced_psf


def compute_dead_loads(
    slab_thickness_in: float,
    concrete_density_pcf: float,
    sdl_psf: float,
    tributary_width_ft: float,
    beam_web_width_in: float,
    beam_depth_in: float
) -> Tuple[float, float, float]:
    """
    Compute dead loads.
    
    Returns:
        (wD_slab_plf, wD_beam_plf, wD_total_plf)
    """
    # Slab dead load (psf)
    d_slab_psf = concrete_density_pcf * (slab_thickness_in / 12.0)
    
    # Total dead load from slab + SDL (psf)
    d_total_psf = d_slab_psf + sdl_psf
    
    # Convert to line load from slab+SDL
    wd_slab_plf = d_total_psf * tributary_width_ft
    
    # Beam self-weight
    beam_area_ft2 = (beam_web_width_in / 12.0) * (beam_depth_in / 12.0)
    wd_beam_plf = beam_area_ft2 * concrete_density_pcf
    
    # Total dead line load
    wd_total_plf = wd_slab_plf + wd_beam_plf
    
    return wd_slab_plf, wd_beam_plf, wd_total_plf


def compute_effective_span_ln(
    l_cc_ft: float,
    col_width_ft: float,
    use_effective_span: bool = True,
    effective_depth_ft: Optional[float] = None,
    beam_depth_ft: Optional[float] = None,
    d_factor: float = 0.9
) -> float:
    """
    Compute effective span length ln per ACI.
    
    If use_effective_span is False, returns L_cc directly.
    Otherwise: ln = min(L_cc, L_clear + d)
    
    If effective_depth_ft is not provided, estimates d = d_factor * beam_depth
    """
    if not use_effective_span:
        return l_cc_ft
    
    l_clear_ft = l_cc_ft - col_width_ft
    
    if effective_depth_ft is None:
        if beam_depth_ft is not None:
            # Estimate d from beam depth
            effective_depth_ft = d_factor * beam_depth_ft
        else:
            # If no beam depth provided, use L_clear only
            effective_depth_ft = 0.0
    
    ln = min(l_cc_ft, l_clear_ft + effective_depth_ft)
    return ln


def aci_moment_from_coeff(wu_plf: float, ln_ft: float, denominator: float) -> float:
    """
    Compute moment using ACI coefficient: M = (wu * ln^2) / denominator
    
    Returns moment in lb-ft
    """
    moment_lbft = (wu_plf * ln_ft ** 2) / denominator
    return moment_lbft


def end_shears_from_end_moments(
    w_plf: float,
    l_ft: float,
    m_left_lbft: float,
    m_right_lbft: float
) -> Tuple[float, float]:
    """
    Compute end shears from uniform load and end moments.
    
    Returns:
        (V_left, V_right) in lb
    """
    v_left = w_plf * l_ft / 2.0 + (m_left_lbft - m_right_lbft) / l_ft
    v_right = w_plf * l_ft / 2.0 + (m_right_lbft - m_left_lbft) / l_ft
    return v_left, v_right


def moment_at_face(
    m_support_lbft: float,
    v_support_lb: float,
    col_width_ft: float
) -> float:
    """
    Compute moment at face of column.
    
    M_face = M_support - V_support * a
    where a = col_width / 2
    """
    a_ft = col_width_ft / 2.0
    m_face_lbft = m_support_lbft - v_support_lb * a_ft
    return m_face_lbft


def process_beam(
    beam_name: str,
    span_lengths_cc_ft: List[float],
    tributary_widths_ft: List[float],
    config: Dict
) -> None:
    """
    Process a single beam line and output results.
    """
    print("\n" + "="*80)
    print(f"BEAM LINE: {beam_name}")
    print("="*80)
    
    # Extract configuration
    slab_thickness_in = config.get("slab_thickness_in", 6.0)
    concrete_density_pcf = config.get("concrete_density_pcf", 150.0)
    sdl_psf = config.get("sdl_psf", 20.0)
    l0_psf = config.get("live_load_psf", 60.0)
    beam_web_width_in = config.get("beam_web_width_in", 12.0)
    beam_depth_in = config.get("beam_depth_in", 24.0)
    col_width_in = config.get("column_width_in", 18.0)
    col_width_ft = col_width_in / 12.0
    dead_factor = config.get("dead_factor", 1.2)
    live_factor = config.get("live_factor", 1.6)
    use_effective_span = config.get("use_effective_span_ln", True)
    effective_depth_in = config.get("effective_depth_d_in")
    effective_depth_ft = effective_depth_in / 12.0 if effective_depth_in else None
    beam_depth_ft = beam_depth_in / 12.0
    d_factor = config.get("d_factor", 0.9)
    
    # ACI coefficient denominators
    coeffs = config.get("aci_coefficients", {})
    end_span_coeffs = coeffs.get("end_span", {})
    interior_span_coeffs = coeffs.get("interior_span", {})
    
    neg_ext_den = end_span_coeffs.get("neg_exterior_support_den", 24)
    neg_int_den = end_span_coeffs.get("neg_first_interior_support_den", 9)
    pos_end_den = end_span_coeffs.get("pos_midspan_den", 14)
    pos_int_den = interior_span_coeffs.get("pos_midspan_den", 16)
    neg_support_den = interior_span_coeffs.get("neg_support_den_optional", 11)
    
    # Check ACI applicability
    assume_uniform = config.get("assume_uniform_loads", True)
    assume_prismatic = config.get("assume_prismatic_member", True)
    span_ratio_limit = config.get("span_ratio_limit", 1.20)
    
    is_applicable, reasons = check_aci_coeff_applicability(
        len(span_lengths_cc_ft),
        span_lengths_cc_ft,
        assume_uniform,
        assume_prismatic,
        span_ratio_limit
    )
    
    print("\nACI Coefficient Applicability Check:")
    print("  Result: " + ("YES" if is_applicable else "NO"))
    for reason in reasons:
        print(f"    {reason}")
    
    if not is_applicable:
        print("\n  WARNING: ACI coefficients may not be applicable!")
    
    # Live load reduction parameters
    use_live_reduction = config.get("use_live_load_reduction", False)
    ll_reduction_params = config.get("live_load_reduction", {})
    
    # Process each span
    num_spans = len(span_lengths_cc_ft)
    results = []
    
    for i in range(num_spans):
        span_name = f"Span {i+1}"
        l_cc_ft = span_lengths_cc_ft[i]
        trib_width_ft = tributary_widths_ft[i]
        
        # Compute effective span
        ln_ft = compute_effective_span_ln(
            l_cc_ft, col_width_ft, use_effective_span, 
            effective_depth_ft, beam_depth_ft, d_factor
        )
        
        # Compute dead loads
        wd_slab, wd_beam, wd_total = compute_dead_loads(
            slab_thickness_in, concrete_density_pcf, sdl_psf,
            trib_width_ft, beam_web_width_in, beam_depth_in
        )
        
        # Live load reduction
        if use_live_reduction:
            # Get KLL for this case (default if not specified)
            case_key = f"span_{i+1}"
            case_params = ll_reduction_params.get(case_key, {})
            kll = case_params.get("KLL", 2.0)  # Default for beams
            at_override = case_params.get("At_override_ft2")
            at_ft2 = at_override if at_override else (trib_width_ft * l_cc_ft)
            
            min_frac = case_params.get("min_reduction_fraction")
            max_frac = case_params.get("max_reduction_fraction")
            
            l_reduced_psf = compute_live_load_reduction(
                l0_psf, at_ft2, kll, min_frac, max_frac
            )
        else:
            l_reduced_psf = l0_psf
            at_ft2 = trib_width_ft * l_cc_ft
            kll = None
        
        # Line loads
        wl_plf = l_reduced_psf * trib_width_ft
        wu_plf = dead_factor * wd_total + live_factor * wl_plf
        
        # Store results
        results.append({
            "span_name": span_name,
            "l_cc_ft": l_cc_ft,
            "ln_ft": ln_ft,
            "trib_width_ft": trib_width_ft,
            "wd_total_plf": wd_total,
            "wl_plf": wl_plf,
            "wu_plf": wu_plf,
            "l_reduced_psf": l_reduced_psf,
            "at_ft2": at_ft2,
            "kll": kll
        })
    
    # Print load summary
    print("\n" + "-"*80)
    print("LOAD SUMMARY")
    print("-"*80)
    for res in results:
        print(f"\n{res['span_name']}:")
        print(f"  L_cc = {res['l_cc_ft']:.2f} ft")
        print(f"  ln = {res['ln_ft']:.2f} ft")
        print(f"  Tributary width = {res['trib_width_ft']:.2f} ft")
        print(f"  wD_total = {res['wd_total_plf']:.2f} plf")
        print(f"  wL = {res['wl_plf']:.2f} plf")
        print(f"  wu = {res['wu_plf']:.2f} plf")
        if use_live_reduction:
            print(f"  Live load reduction:")
            print(f"    At = {res['at_ft2']:.2f} ft²")
            print(f"    KLL = {res['kll']:.2f}")
            print(f"    L0 = {l0_psf:.2f} psf")
            print(f"    L_reduced = {res['l_reduced_psf']:.2f} psf")
    
    # Compute design moments
    print("\n" + "-"*80)
    print("DESIGN MOMENTS (Factored)")
    print("-"*80)
    print(f"\nACI Coefficients Used:")
    print(f"  End span - Exterior negative: 1/{neg_ext_den}")
    print(f"  End span - First interior negative: 1/{neg_int_den}")
    print(f"  End span - Positive midspan: 1/{pos_end_den}")
    print(f"  Interior span - Positive midspan: 1/{pos_int_den}")
    print(f"  Interior support negative: 1/{neg_support_den}")
    
    # End span (first span)
    if num_spans > 0:
        end_span = results[0]
        print(f"\n--- END SPAN ({end_span['span_name']}) ---")
        
        # Negative at exterior support
        m_neg_ext_lbft = aci_moment_from_coeff(end_span['wu_plf'], end_span['ln_ft'], neg_ext_den)
        print(f"  M_neg at exterior support = {m_neg_ext_lbft/1000:.2f} kip-ft")
        
        # Positive at midspan
        m_pos_end_lbft = aci_moment_from_coeff(end_span['wu_plf'], end_span['ln_ft'], pos_end_den)
        print(f"  M_pos at midspan = {m_pos_end_lbft/1000:.2f} kip-ft")
        
        # Negative at first interior support (from end span side)
        m_neg_int_end_lbft = aci_moment_from_coeff(end_span['wu_plf'], end_span['ln_ft'], neg_int_den)
        print(f"  M_neg at first interior support (from end span) = {m_neg_int_end_lbft/1000:.2f} kip-ft")
    
    # Interior spans
    for i in range(1, num_spans):
        interior_span = results[i]
        print(f"\n--- INTERIOR SPAN ({interior_span['span_name']}) ---")
        
        # Positive at midspan
        m_pos_int_lbft = aci_moment_from_coeff(interior_span['wu_plf'], interior_span['ln_ft'], pos_int_den)
        print(f"  M_pos at midspan = {m_pos_int_lbft/1000:.2f} kip-ft")
    
    # Interior support moments (at face of column)
    if num_spans >= 2:
        print(f"\n--- INTERIOR SUPPORT MOMENTS (at face of column) ---")
        
        # For each interior support
        for i in range(1, num_spans):
            support_name = f"Support {i+1}"
            left_span = results[i-1]
            right_span = results[i]
            
            # Compute negative moments from both sides
            # Left side: from end span (if first interior) or interior span
            if i == 1:
                # First interior support: use end span coefficient
                m_neg_left_lbft = aci_moment_from_coeff(left_span['wu_plf'], left_span['ln_ft'], neg_int_den)
            else:
                # Interior support: use interior support coefficient
                m_neg_left_lbft = aci_moment_from_coeff(left_span['wu_plf'], left_span['ln_ft'], neg_support_den)
            
            # Right side: could be interior span or end span
            if i == num_spans - 1:
                # Last interior support: right span is end span
                m_neg_right_lbft = aci_moment_from_coeff(right_span['wu_plf'], right_span['ln_ft'], neg_int_den)
            else:
                # Interior support: use interior support coefficient
                m_neg_right_lbft = aci_moment_from_coeff(right_span['wu_plf'], right_span['ln_ft'], neg_support_den)
            
            # Take the larger magnitude (more negative)
            if abs(m_neg_left_lbft) >= abs(m_neg_right_lbft):
                m_support_lbft = m_neg_left_lbft
                governing_span = left_span
                governing_side = "left"
                # For left side, need right end moment
                m_left = 0.0 if i == 1 else aci_moment_from_coeff(
                    results[i-2]['wu_plf'], results[i-2]['ln_ft'], 
                    neg_int_den if i == 2 else neg_support_den
                )
                m_right = m_support_lbft
            else:
                m_support_lbft = m_neg_right_lbft
                governing_span = right_span
                governing_side = "right"
                # For right side, need left end moment (this support) and right end moment
                m_left = m_support_lbft
                if i < num_spans - 1:
                    m_right = aci_moment_from_coeff(
                        results[i+1]['wu_plf'], results[i+1]['ln_ft'],
                        neg_support_den
                    )
                else:
                    m_right = 0.0  # End support
            
            # Compute end shears to get V at support
            if governing_side == "left":
                # From left span: V_right is the shear at this support
                _, v_support_lb = end_shears_from_end_moments(
                    governing_span['wu_plf'], governing_span['ln_ft'], m_left, m_right
                )
            else:
                # From right span: V_left is the shear at this support
                v_support_lb, _ = end_shears_from_end_moments(
                    governing_span['wu_plf'], governing_span['ln_ft'], m_left, m_right
                )
            
            # Moment at face of column
            m_face_lbft = moment_at_face(m_support_lbft, v_support_lb, col_width_ft)
            
            print(f"\n  {support_name}:")
            print(f"    M_neg at centerline = {m_support_lbft/1000:.2f} kip-ft")
            print(f"    V at support = {v_support_lb/1000:.2f} kips")
            print(f"    M_neg at face of column = {m_face_lbft/1000:.2f} kip-ft")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Reinforced Concrete Continuous Beam Design Moment Calculator"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="JSON input file (if not provided, uses default example)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Use default configuration
        print("No input file provided. Using default configuration.")
        print("Usage: python beam_design.py input.json")
        print("\nUsing built-in example configuration...\n")
        config = get_default_config()
    
    # Process both beams
    beam_a_b_c_d = config.get("beam_A_B_C_D", {})
    beam_e_f_g_h = config.get("beam_E_F_G_H", {})
    
    if beam_a_b_c_d:
        process_beam(
            "A-B-C-D",
            beam_a_b_c_d.get("span_lengths_cc_ft", []),
            beam_a_b_c_d.get("tributary_widths_ft", []),
            config
        )
    
    if beam_e_f_g_h:
        process_beam(
            "E-F-G-H",
            beam_e_f_g_h.get("span_lengths_cc_ft", []),
            beam_e_f_g_h.get("tributary_widths_ft", []),
            config
        )


def get_default_config() -> Dict:
    """Return a default configuration for testing."""
    return {
        "slab_thickness_in": 6.0,
        "concrete_density_pcf": 150.0,
        "sdl_psf": 20.0,
        "live_load_psf": 60.0,
        "beam_web_width_in": 12.0,
        "beam_depth_in": 24.0,
        "column_width_in": 18.0,
        "dead_factor": 1.2,
        "live_factor": 1.6,
        "use_effective_span_ln": True,
        "assume_uniform_loads": True,
        "assume_prismatic_member": True,
        "span_ratio_limit": 1.20,
        "use_live_load_reduction": True,
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
            "span_1": {"KLL": 2.0},
            "span_2": {"KLL": 2.0},
            "span_3": {"KLL": 2.0}
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


if __name__ == "__main__":
    main()
