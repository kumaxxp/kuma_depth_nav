#!/usr/bin/env python3
"""
深度カメラアプリケーションのエントリポイント
このスクリプトは、リファクタリングされたアプリケーションを起動します
"""
import uvicorn

def main():
    try:
        print("リファクタリングされた深度カメラアプリケーションを起動します...")
        uvicorn.run("app.main:app", host="0.0.0.0", port=8888, reload=False)
    except KeyboardInterrupt:
        print("Ctrl+Cが押されました。アプリケーションを終了します。")
    except Exception as e:
        print(f"予期せぬエラーでアプリケーションが終了しました: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
