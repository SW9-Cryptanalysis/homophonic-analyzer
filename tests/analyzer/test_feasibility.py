from src.analyzer.feasibility import is_feasible, calculate_target_range

def test_is_feasible():
    """Test feasibility at the exact threshold boundary"""
    # Feasibility threshold is 1,000,000,000
    
    assert is_feasible(30, 15)   # C(30,15) = 155,117,520
    assert is_feasible(28, 14)   # C(28,14) = 40,116,600
    
    assert not is_feasible(35, 17)  # C(35,17) = 4,537,567,650
    assert not is_feasible(40, 20)  # C(40,20) = 137,846,528,820

def test_calculate_target_range():
    assert calculate_target_range(114, 0.01) == (1, 2)
    assert calculate_target_range(156, 0.12702) == (19, 21)
    assert calculate_target_range(56, 0.05) == (2, 3)

def test_calculate_target_range_edge_cases():
    """Test target range calculation for edge cases"""
    
    # Very small frequencies
    min_count, max_count = calculate_target_range(1000, 0.0001)
    assert min_count >= 1
    
    # Very large frequencies (should be capped appropriately)
    min_count, max_count = calculate_target_range(100, 0.5)
    assert min_count < max_count
    assert max_count <= 100
    
    min_count, max_count = calculate_target_range(100, 0.1)
    assert min_count <= 10 <= max_count