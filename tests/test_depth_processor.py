import unittest
import numpy as np
import cv2
import logging

from depth_processor import convert_to_absolute_depth, create_depth_visualization

# Logger設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
    def test_depth_visualization(self):
        """深度可視化のテスト"""
        try:
            # テスト用の深度マップを生成
            dummy_depth = np.zeros((1, 256, 384, 1), dtype=np.float32)
            for y in range(256):
                # 下部ほど近く（値が大きい）
                value = 0.1 + 0.8 * (y / 255)
                dummy_depth[0, y, :, 0] = value
                
            # 可視化のテスト
            from depth_processor.visualization import create_depth_visualization
            test_image = create_depth_visualization(dummy_depth, (480, 640))
            
            # 結果を確認
            if test_image is not None and test_image.shape[0] > 0:
                logger.info(f"Visualization test successful. Output shape: {test_image.shape}")
                
                # ファイルに保存
                cv2.imwrite("test_depth_viz.jpg", test_image)
                logger.info("Test visualization saved to: test_depth_viz.jpg")
                return True
            else:
                logger.error("Visualization failed to produce valid output")
                return False
        except Exception as e:
            logger.error(f"Visualization test error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

# アプリケーション起動前にテスト実行
try:
    test_result = TestDepthProcessor().test_depth_visualization()
    if test_result:
        logger.info("Visualization test passed!")
    else:
        logger.warning("Visualization test failed!")
except Exception as e:
    logger.error(f"Error during test: {e}")

if __name__ == '__main__':
    unittest.main()