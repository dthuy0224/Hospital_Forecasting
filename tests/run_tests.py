#!/usr/bin/env python3
"""
Test runner for Hospital Forecasting Project
"""

import unittest
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def run_all_tests():
    """Run all test suites"""
    
    print("🧪 Hospital Forecasting Project - Test Suite")
    print("=" * 60)
    
    # Import test modules
    from tests.test_data_processing import (
        TestDataPreprocessor, 
        TestAdvancedFeatureEngineer, 
        TestDataProcessingIntegration
    )
    
    from tests.test_models import (
        TestHospitalDemandForecaster,
        TestModelComparator,
        TestModelOptimizer,
        TestModelsIntegration
    )
    
    from tests.test_dashboard import (
        TestDashboardFunctions,
        TestDashboardIntegration
    )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    print("📊 Adding Data Processing Tests...")
    test_suite.addTest(unittest.makeSuite(TestDataPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestAdvancedFeatureEngineer))
    test_suite.addTest(unittest.makeSuite(TestDataProcessingIntegration))
    
    print("🤖 Adding Model Tests...")
    test_suite.addTest(unittest.makeSuite(TestHospitalDemandForecaster))
    test_suite.addTest(unittest.makeSuite(TestModelComparator))
    test_suite.addTest(unittest.makeSuite(TestModelOptimizer))
    test_suite.addTest(unittest.makeSuite(TestModelsIntegration))
    
    print("📈 Adding Dashboard Tests...")
    test_suite.addTest(unittest.makeSuite(TestDashboardFunctions))
    test_suite.addTest(unittest.makeSuite(TestDashboardIntegration))
    
    # Run tests
    print("\n🚀 Running Tests...")
    print("-" * 60)
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests:     {total_tests}")
    print(f"✅ Passed:       {passed}")
    print(f"❌ Failed:       {failures}")
    print(f"💥 Errors:       {errors}")
    print(f"⏭️  Skipped:      {skipped}")
    print(f"⏱️  Duration:     {end_time - start_time:.2f}s")
    print(f"📈 Success Rate: {(passed / total_tests * 100):.1f}%")
    
    # Print detailed failure/error information
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"• {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n💥 ERRORS ({errors}):")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"• {test}: {traceback.split('Error:')[-1].strip()}")
    
    # Print recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if failures > 0 or errors > 0:
        print("• Fix failing tests before proceeding to production")
        print("• Review error messages and update test data if needed")
        print("• Consider adding more edge case tests")
    else:
        print("• All tests passed! Ready for production deployment")
        print("• Consider adding performance tests for large datasets")
        print("• Add integration tests with real data sources")
    
    print("=" * 60)
    
    # Return exit code
    return 0 if (failures == 0 and errors == 0) else 1

def run_specific_test_suite(suite_name):
    """Run a specific test suite"""
    
    print(f"🧪 Running {suite_name} Tests...")
    print("=" * 60)
    
    if suite_name.lower() == 'data':
        from tests.test_data_processing import (
            TestDataPreprocessor, 
            TestAdvancedFeatureEngineer, 
            TestDataProcessingIntegration
        )
        test_suite = unittest.TestSuite()
        test_suite.addTest(unittest.makeSuite(TestDataPreprocessor))
        test_suite.addTest(unittest.makeSuite(TestAdvancedFeatureEngineer))
        test_suite.addTest(unittest.makeSuite(TestDataProcessingIntegration))
        
    elif suite_name.lower() == 'models':
        from tests.test_models import (
            TestHospitalDemandForecaster,
            TestModelComparator,
            TestModelOptimizer,
            TestModelsIntegration
        )
        test_suite = unittest.TestSuite()
        test_suite.addTest(unittest.makeSuite(TestHospitalDemandForecaster))
        test_suite.addTest(unittest.makeSuite(TestModelComparator))
        test_suite.addTest(unittest.makeSuite(TestModelOptimizer))
        test_suite.addTest(unittest.makeSuite(TestModelsIntegration))
        
    elif suite_name.lower() == 'dashboard':
        from tests.test_dashboard import (
            TestDashboardFunctions,
            TestDashboardIntegration
        )
        test_suite = unittest.TestSuite()
        test_suite.addTest(unittest.makeSuite(TestDashboardFunctions))
        test_suite.addTest(unittest.makeSuite(TestDashboardIntegration))
    
    elif suite_name.lower() == 'integration':
        from tests.test_integration import (
            TestEndToEndPipeline,
            TestPerformanceIntegration
        )
        test_suite = unittest.TestSuite()
        test_suite.addTest(unittest.makeSuite(TestEndToEndPipeline))
        test_suite.addTest(unittest.makeSuite(TestPerformanceIntegration))
        
    else:
        print(f"❌ Unknown test suite: {suite_name}")
        print("Available suites: data, models, dashboard, integration")
        return 1
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    passed = result.testsRun - len(result.failures) - len(result.errors)
    success_rate = (passed / result.testsRun * 100) if result.testsRun > 0 else 0
    
    print(f"\n📊 {suite_name.upper()} Tests Summary:")
    print(f"✅ Passed: {passed}/{result.testsRun} ({success_rate:.1f}%)")
    
    return 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Hospital Forecasting tests')
    parser.add_argument('--suite', '-s', 
                       choices=['data', 'models', 'dashboard', 'integration', 'all'],
                       default='all',
                       help='Test suite to run (default: all)')
    
    args = parser.parse_args()
    
    if args.suite == 'all':
        exit_code = run_all_tests()
    else:
        exit_code = run_specific_test_suite(args.suite)
    
    sys.exit(exit_code)
