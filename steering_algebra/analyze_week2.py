"""
Analyze Week 2 results and generate summary statistics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def analyze_week2_results(results_path: Path) -> Dict:
    """Analyze Week 2 composition results."""
    
    with open(results_path) as f:
        results = json.load(f)
    
    analysis = {}
    
    # 1. Additive Composition Analysis
    if results['additive_composition']:
        additive = results['additive_composition']
        
        success_rates = [r['composition_success_rate'] for r in additive]
        similarities = [abs(r['cosine_similarity']) for r in additive]
        
        # Categorize by geometry
        orthogonal = [r for r in additive if abs(r['cosine_similarity']) < 0.2]
        other = [r for r in additive if abs(r['cosine_similarity']) >= 0.2]
        
        analysis['additive'] = {
            'n_pairs_tested': len(additive),
            'mean_success_rate': float(np.mean(success_rates)),
            'successful_pairs': sum(1 for r in success_rates if r > 0.6),
            'orthogonal_success': float(np.mean([r['composition_success_rate'] for r in orthogonal])) if orthogonal else None,
            'other_success': float(np.mean([r['composition_success_rate'] for r in other])) if other else None,
            'best_pair': max(additive, key=lambda x: x['composition_success_rate']),
            'worst_pair': min(additive, key=lambda x: x['composition_success_rate'])
        }
    
    # 2. Opposing Composition Analysis
    if results['opposing_composition']:
        opposing = results['opposing_composition']
        
        cancels_count = sum(1 for r in opposing if r['cancels_out'])
        
        analysis['opposing'] = {
            'n_pairs_tested': len(opposing),
            'cancels_out_count': cancels_count,
            'cancellation_rate': cancels_count / len(opposing) if opposing else 0,
            'mean_similarity': float(np.mean([r['cosine_similarity'] for r in opposing]))
        }
    
    # 3. Arithmetic Composition Analysis
    if results['arithmetic_composition']:
        arithmetic = results['arithmetic_composition']
        
        works_count = sum(1 for r in arithmetic if r['arithmetic_works'])
        differences = [r['difference'] for r in arithmetic]
        
        analysis['arithmetic'] = {
            'n_pairs_tested': len(arithmetic),
            'works_count': works_count,
            'success_rate': works_count / len(arithmetic) if arithmetic else 0,
            'mean_difference': float(np.mean(differences)),
            'median_difference': float(np.median(differences))
        }
    
    # 4. Geometry Analysis
    if results['geometry_analysis']:
        analysis['geometry'] = results['geometry_analysis']
    
    return analysis


def print_summary(analysis: Dict):
    """Print a human-readable summary."""
    
    print("\n" + "="*60)
    print("WEEK 2 RESULTS SUMMARY")
    print("="*60)
    
    # Additive Composition
    if 'additive' in analysis:
        a = analysis['additive']
        print("\n1. ADDITIVE COMPOSITION (A + B)")
        print("-" * 40)
        print(f"Pairs tested: {a['n_pairs_tested']}")
        print(f"Mean success rate: {a['mean_success_rate']:.1%}")
        print(f"Successful pairs (>60%): {a['successful_pairs']}/{a['n_pairs_tested']}")
        
        if a['orthogonal_success'] is not None:
            print(f"\nOrthogonal pairs: {a['orthogonal_success']:.1%} success")
        if a['other_success'] is not None:
            print(f"Non-orthogonal pairs: {a['other_success']:.1%} success")
        
        print(f"\nBest pair: {a['best_pair']['concept_a']} + {a['best_pair']['concept_b']}")
        print(f"  Success rate: {a['best_pair']['composition_success_rate']:.1%}")
        print(f"Worst pair: {a['worst_pair']['concept_a']} + {a['worst_pair']['concept_b']}")
        print(f"  Success rate: {a['worst_pair']['composition_success_rate']:.1%}")
    
    # Opposing Composition
    if 'opposing' in analysis:
        o = analysis['opposing']
        print("\n2. OPPOSING COMPOSITION (A + (-A))")
        print("-" * 40)
        print(f"Pairs tested: {o['n_pairs_tested']}")
        print(f"Cancellation rate: {o['cancellation_rate']:.1%}")
        print(f"Mean similarity: {o['mean_similarity']:.3f}")
    
    # Arithmetic
    if 'arithmetic' in analysis:
        ar = analysis['arithmetic']
        print("\n3. VECTOR ARITHMETIC ((A+B)-A = B)")
        print("-" * 40)
        print(f"Pairs tested: {ar['n_pairs_tested']}")
        print(f"Success rate: {ar['success_rate']:.1%}")
        print(f"Mean difference: {ar['mean_difference']:.3f}")
    
    # Geometry
    if 'geometry' in analysis:
        g = analysis['geometry']
        print("\n4. GEOMETRY vs COMPOSITION")
        print("-" * 40)
        if 'correlation' in g:
            print(f"Correlation (similarity vs success): {g['correlation']:.3f}")
        if g.get('orthogonal_success'):
            print(f"Orthogonal pairs: {g['orthogonal_success']:.1%} success (n={g['n_orthogonal']})")
        if g.get('aligned_success'):
            print(f"Aligned pairs: {g['aligned_success']:.1%} success (n={g['n_aligned']})")
        if g.get('opposing_success'):
            print(f"Opposing pairs: {g['opposing_success']:.1%} success (n={g['n_opposing']})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="outputs/week2")
    args = parser.parse_args()
    
    results_path = Path(args.results_dir) / "composition_results.json"
    
    if not results_path.exists():
        print(f"Error: Results not found at {results_path}")
        print("Run week 2 experiments first: python run_week2.py")
        return
    
    analysis = analyze_week2_results(results_path)
    print_summary(analysis)
    
    # Save analysis
    output_path = Path(args.results_dir) / "analysis_summary.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n\nDetailed analysis saved to: {output_path}")


if __name__ == "__main__":
    main()