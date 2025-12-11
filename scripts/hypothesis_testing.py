"""
Hypothesis Testing Module for Insurance Analytics
Implements A/B testing for risk differences across various segments
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HypothesisTester:
    """Class for conducting hypothesis tests on insurance data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize HypothesisTester
        
        Args:
            df: DataFrame with insurance data
        """
        self.df = df.copy()
        self.results = {}
        
    def calculate_metrics(self, group_data: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics for a group
        
        Args:
            group_data: DataFrame for specific group
            
        Returns:
            Dictionary with calculated metrics
        """
        total_policies = len(group_data)
        policies_with_claims = (group_data['TotalClaims'] > 0).sum()
        
        # Claim Frequency: proportion of policies with at least one claim
        claim_frequency = policies_with_claims / total_policies if total_policies > 0 else 0
        
        # Claim Severity: average claim amount for policies with claims
        claims_data = group_data[group_data['TotalClaims'] > 0]['TotalClaims']
        claim_severity = claims_data.mean() if len(claims_data) > 0 else 0
        
        # Margin: TotalPremium - TotalClaims
        margin = group_data['TotalPremium'].sum() - group_data['TotalClaims'].sum()
        avg_margin = margin / total_policies if total_policies > 0 else 0
        
        # Loss Ratio
        total_premium = group_data['TotalPremium'].sum()
        total_claims = group_data['TotalClaims'].sum()
        loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        
        return {
            'n_policies': total_policies,
            'claim_frequency': claim_frequency,
            'claim_severity': claim_severity,
            'avg_margin': avg_margin,
            'total_margin': margin,
            'loss_ratio': loss_ratio,
            'total_premium': total_premium,
            'total_claims': total_claims
        }
    
    def chi_squared_test(self, group_a: pd.DataFrame, group_b: pd.DataFrame,
                        group_a_name: str, group_b_name: str) -> Dict:
        """
        Perform chi-squared test for claim frequency differences
        
        Args:
            group_a: First group data
            group_b: Second group data
            group_a_name: Name of first group
            group_b_name: Name of second group
            
        Returns:
            Dictionary with test results
        """
        # Create contingency table for claims (has claim vs no claim)
        a_has_claims = (group_a['TotalClaims'] > 0).sum()
        a_no_claims = len(group_a) - a_has_claims
        b_has_claims = (group_b['TotalClaims'] > 0).sum()
        b_no_claims = len(group_b) - b_has_claims
        
        contingency_table = np.array([
            [a_has_claims, a_no_claims],
            [b_has_claims, b_no_claims]
        ])
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        metrics_a = self.calculate_metrics(group_a)
        metrics_b = self.calculate_metrics(group_b)
        
        return {
            'test_type': 'Chi-Squared Test',
            'group_a': group_a_name,
            'group_b': group_b_name,
            'group_a_metrics': metrics_a,
            'group_b_metrics': metrics_b,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'reject_null': p_value < 0.05,
            'significance_level': 0.05
        }
    
    def t_test(self, group_a: pd.DataFrame, group_b: pd.DataFrame,
              group_a_name: str, group_b_name: str, 
              metric: str = 'margin') -> Dict:
        """
        Perform t-test for metric differences between groups
        
        Args:
            group_a: First group data
            group_b: Second group data
            group_a_name: Name of first group
            group_b_name: Name of second group
            metric: Metric to test ('margin', 'claims', 'premium')
            
        Returns:
            Dictionary with test results
        """
        if metric == 'margin':
            a_values = group_a['TotalPremium'] - group_a['TotalClaims']
            b_values = group_b['TotalPremium'] - group_b['TotalClaims']
        elif metric == 'claims':
            a_values = group_a['TotalClaims']
            b_values = group_b['TotalClaims']
        elif metric == 'premium':
            a_values = group_a['TotalPremium']
            b_values = group_b['TotalPremium']
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Perform independent samples t-test
        t_statistic, p_value = stats.ttest_ind(a_values, b_values, equal_var=False)
        
        metrics_a = self.calculate_metrics(group_a)
        metrics_b = self.calculate_metrics(group_b)
        
        return {
            'test_type': 'Independent T-Test',
            'metric_tested': metric,
            'group_a': group_a_name,
            'group_b': group_b_name,
            'group_a_metrics': metrics_a,
            'group_b_metrics': metrics_b,
            'group_a_mean': a_values.mean(),
            'group_b_mean': b_values.mean(),
            'mean_difference': a_values.mean() - b_values.mean(),
            't_statistic': t_statistic,
            'p_value': p_value,
            'reject_null': p_value < 0.05,
            'significance_level': 0.05
        }
    
    def anova_test(self, groups: Dict[str, pd.DataFrame], 
                   metric: str = 'margin') -> Dict:
        """
        Perform ANOVA test for multiple groups
        
        Args:
            groups: Dictionary mapping group names to DataFrames
            metric: Metric to test
            
        Returns:
            Dictionary with test results
        """
        group_values = []
        group_names = []
        
        for name, data in groups.items():
            if metric == 'margin':
                values = data['TotalPremium'] - data['TotalClaims']
            elif metric == 'claims':
                values = data['TotalClaims']
            elif metric == 'premium':
                values = data['TotalPremium']
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            group_values.append(values)
            group_names.append(name)
        
        f_statistic, p_value = stats.f_oneway(*group_values)
        
        group_metrics = {name: self.calculate_metrics(data) 
                        for name, data in groups.items()}
        
        return {
            'test_type': 'ANOVA',
            'metric_tested': metric,
            'groups': group_names,
            'n_groups': len(groups),
            'group_metrics': group_metrics,
            'f_statistic': f_statistic,
            'p_value': p_value,
            'reject_null': p_value < 0.05,
            'significance_level': 0.05
        }
    
    def test_provinces(self) -> Dict:
        """
        Test H₀: There are no risk differences across provinces
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing risk differences across provinces")
        
        if 'Province' not in self.df.columns:
            return {'error': 'Province column not found'}
        
        # Get all provinces
        provinces = self.df['Province'].dropna().unique()
        
        if len(provinces) < 2:
            return {'error': 'Need at least 2 provinces for comparison'}
        
        # Create groups dictionary
        groups = {province: self.df[self.df['Province'] == province] 
                 for province in provinces}
        
        # Perform ANOVA for margin differences
        result = self.anova_test(groups, metric='margin')
        
        # Also test claim frequency with chi-squared (pairwise for top 2 provinces)
        province_sizes = {p: len(g) for p, g in groups.items()}
        top_2_provinces = sorted(province_sizes.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if len(top_2_provinces) == 2:
            prov_a, prov_b = top_2_provinces[0][0], top_2_provinces[1][0]
            chi_result = self.chi_squared_test(
                groups[prov_a], groups[prov_b], prov_a, prov_b
            )
            result['pairwise_chi_squared'] = chi_result
        
        self.results['provinces'] = result
        return result
    
    def test_postal_codes(self, sample_size: int = 20) -> Dict:
        """
        Test H₀: There are no risk differences between postal codes
        
        Args:
            sample_size: Number of top postal codes to compare
            
        Returns:
            Dictionary with test results
        """
        logger.info("Testing risk differences between postal codes")
        
        if 'PostalCode' not in self.df.columns:
            return {'error': 'PostalCode column not found'}
        
        # Get top postal codes by policy count
        postal_counts = self.df['PostalCode'].value_counts()
        top_postal_codes = postal_counts.head(sample_size).index
        
        # Create groups
        groups = {str(pc): self.df[self.df['PostalCode'] == pc] 
                 for pc in top_postal_codes}
        
        # Perform ANOVA for margin differences
        result = self.anova_test(groups, metric='margin')
        
        # Also do pairwise test for top 2
        if len(top_postal_codes) >= 2:
            pc_a, pc_b = top_postal_codes[0], top_postal_codes[1]
            t_result = self.t_test(
                groups[str(pc_a)], groups[str(pc_b)], 
                str(pc_a), str(pc_b), metric='margin'
            )
            result['pairwise_t_test'] = t_result
        
        self.results['postal_codes'] = result
        return result
    
    def test_gender(self) -> Dict:
        """
        Test H₀: There is no significant risk difference between Women and Men
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing risk differences between genders")
        
        if 'Gender' not in self.df.columns:
            return {'error': 'Gender column not found'}
        
        # Filter for Male and Female only
        gender_data = self.df[self.df['Gender'].isin(['Male', 'Female', 'M', 'F', 
                                                        'male', 'female', 'MALE', 'FEMALE'])]
        
        # Standardize gender values
        gender_data = gender_data.copy()
        gender_data['Gender'] = gender_data['Gender'].str.upper().str[0]
        
        male_data = gender_data[gender_data['Gender'] == 'M']
        female_data = gender_data[gender_data['Gender'] == 'F']
        
        if len(male_data) == 0 or len(female_data) == 0:
            return {'error': 'Insufficient data for both genders'}
        
        # Chi-squared test for claim frequency
        chi_result = self.chi_squared_test(male_data, female_data, 'Male', 'Female')
        
        # T-test for margin differences
        t_result = self.t_test(male_data, female_data, 'Male', 'Female', metric='margin')
        
        result = {
            'chi_squared_test': chi_result,
            't_test_margin': t_result
        }
        
        self.results['gender'] = result
        return result
    
    def run_all_tests(self) -> Dict:
        """
        Run all hypothesis tests
        
        Returns:
            Dictionary with all test results
        """
        logger.info("Running all hypothesis tests")
        
        all_results = {
            'provinces': self.test_provinces(),
            'postal_codes': self.test_postal_codes(),
            'gender': self.test_gender()
        }
        
        return all_results
    
    def generate_report(self, results: Optional[Dict] = None) -> str:
        """
        Generate a text report of hypothesis test results
        
        Args:
            results: Test results dictionary (uses self.results if None)
            
        Returns:
            Formatted report string
        """
        if results is None:
            results = self.results
        
        report = ["=" * 80]
        report.append("HYPOTHESIS TESTING REPORT")
        report.append("AlphaCare Insurance Solutions - Risk Analysis")
        report.append("=" * 80)
        report.append("")
        
        for test_name, test_results in results.items():
            report.append(f"\n{test_name.upper().replace('_', ' ')} TEST")
            report.append("-" * 80)
            
            if 'error' in test_results:
                report.append(f"ERROR: {test_results['error']}")
                continue
            
            # Handle different test result structures
            if 'pairwise_chi_squared' in test_results or 'pairwise_t_test' in test_results:
                # ANOVA results
                report.append(f"Test Type: {test_results.get('test_type', 'ANOVA')}")
                report.append(f"Metric: {test_results.get('metric_tested', 'N/A')}")
                report.append(f"F-Statistic: {test_results.get('f_statistic', 0):.4f}")
                report.append(f"P-Value: {test_results.get('p_value', 1):.4f}")
                report.append(f"Decision: {'REJECT H₀' if test_results.get('reject_null') else 'FAIL TO REJECT H₀'}")
            elif 'chi_squared_test' in test_results:
                # Gender test results
                chi = test_results['chi_squared_test']
                report.append(f"Chi-Squared Test (Claim Frequency):")
                report.append(f"  P-Value: {chi['p_value']:.4f}")
                report.append(f"  Decision: {'REJECT H₀' if chi['reject_null'] else 'FAIL TO REJECT H₀'}")
                
                t_test = test_results['t_test_margin']
                report.append(f"\nT-Test (Margin):")
                report.append(f"  P-Value: {t_test['p_value']:.4f}")
                report.append(f"  Decision: {'REJECT H₀' if t_test['reject_null'] else 'FAIL TO REJECT H₀'}")
            
            report.append("")
        
        return "\n".join(report)


def conduct_hypothesis_tests(df: pd.DataFrame) -> Dict:
    """
    Convenience function to conduct all hypothesis tests
    
    Args:
        df: Insurance DataFrame
        
    Returns:
        Dictionary with all test results
    """
    tester = HypothesisTester(df)
    results = tester.run_all_tests()
    report = tester.generate_report(results)
    
    logger.info("\n" + report)
    
    return results


if __name__ == "__main__":
    # Example usage
    pass
