import unittest
import numpy as np

from depth_processor import convert_to_absolute_depth, create_depth_visualization

class TestDepthProcessor(unittest.TestCase):
    
    def test_convert_to_absolute_depth(self):
        """絶対深度変換のテスト"""
        depth_map = np.array([0.0, 0.1, 0.2, 0.5, 1.0])
        scaling_factor = 15.0
        
        result = convert_to_absolute_depth(depth_map, scaling_factor)
        
        # 0.01未満の値は0になることを確認
        self.assertEqual(result[0], 0.0)
        
        # 他の値は正しく変換されることを確認
        self.assertAlmostEqual(result[1], 150.0)  # 15.0 / 0.1
        self.assertAlmostEqual(result[2], 75.0)   # 15.0 / 0.2
        self.assertAlmostEqual(result[3], 30.0)   # 15.0 / 0.5
        self.assertAlmostEqual(result[4], 15.0)   # 15.0 / 1.0
        
    def test_create_depth_visualization_empty_depth(self):
        """空の深度マップに対する可視化テスト"""
        # 空の深度マップ
        depth_map = np.zeros((10, 10))
        
        # エラーなく実行できることを確認
        image = create_depth_visualization(depth_map, (480, 640))
        
        # 何らかの画像が返されることを確認
        self.assertIsNotNone(image)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)

if __name__ == '__main__':
    unittest.main()