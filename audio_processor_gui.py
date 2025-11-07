#!/usr/bin/env python3
"""
音声文字起こし用オーディオ前処理ツール - Webインターフェース版
ブラウザ上で操作できるGUIを提供します。
"""

import os
import sys
import tempfile
import subprocess
import shutil
from pathlib import Path

try:
    import gradio as gr
    import numpy as np
    from moviepy.editor import VideoFileClip
    import noisereduce as nr
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.silence import detect_nonsilent
except ImportError as e:
    print(f"エラー: 必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドで依存ライブラリをインストールしてください:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def check_ffmpeg_availability():
    """FFmpegが使用可能かチェック"""
    result = {
        'available': False,
        'path': None,
        'version': None,
        'message': ''
    }

    # システムのPATHからFFmpegを検索
    ffmpeg_path = shutil.which('ffmpeg')

    if ffmpeg_path:
        result['path'] = ffmpeg_path
        result['available'] = True

        # バージョン情報を取得
        try:
            version_output = subprocess.check_output(
                [ffmpeg_path, '-version'],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=5
            )
            # 最初の行からバージョン情報を抽出
            first_line = version_output.split('\n')[0]
            result['version'] = first_line
            result['message'] = f"✅ FFmpeg使用可能\n📍 パス: {ffmpeg_path}\n📦 {first_line}"
        except Exception as e:
            result['message'] = f"✅ FFmpegが見つかりました\n📍 パス: {ffmpeg_path}\n⚠️ バージョン確認エラー: {e}"
    else:
        # imageio-ffmpegがインストールされているか確認
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            if os.path.exists(ffmpeg_path):
                result['path'] = ffmpeg_path
                result['available'] = True
                result['message'] = f"✅ FFmpeg使用可能（imageio-ffmpeg）\n📍 パス: {ffmpeg_path}"
            else:
                result['message'] = "❌ FFmpegが見つかりません\n\n推奨インストール方法:\n• Windows: https://ffmpeg.org/download.html\n• macOS: brew install ffmpeg\n• Linux: apt install ffmpeg"
        except ImportError:
            result['message'] = "❌ FFmpegが見つかりません\n\n推奨インストール方法:\n• Windows: https://ffmpeg.org/download.html\n• macOS: brew install ffmpeg\n• Linux: apt install ffmpeg"

    return result


def set_ffmpeg_path(custom_path: str):
    """カスタムFFmpegパスを設定"""
    if not custom_path or not custom_path.strip():
        return check_ffmpeg_availability()

    custom_path = custom_path.strip()

    # パスの存在確認
    if not os.path.exists(custom_path):
        return {
            'available': False,
            'path': None,
            'version': None,
            'message': f"❌ 指定されたパスが見つかりません: {custom_path}"
        }

    # 実行可能か確認
    if not os.access(custom_path, os.X_OK):
        return {
            'available': False,
            'path': custom_path,
            'version': None,
            'message': f"❌ 指定されたファイルは実行可能ではありません: {custom_path}"
        }

    result = {
        'available': True,
        'path': custom_path,
        'version': None,
        'message': ''
    }

    # バージョン確認
    try:
        version_output = subprocess.check_output(
            [custom_path, '-version'],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=5
        )
        first_line = version_output.split('\n')[0]
        result['version'] = first_line
        result['message'] = f"✅ カスタムFFmpegを設定しました\n📍 パス: {custom_path}\n📦 {first_line}"

        # 環境変数に設定（moviepyが使用）
        os.environ['FFMPEG_BINARY'] = custom_path
        os.environ['IMAGEIO_FFMPEG_EXE'] = custom_path

        # PyDub用の設定を追加
        AudioSegment.converter = custom_path
        AudioSegment.ffmpeg = custom_path
        AudioSegment.ffprobe = shutil.which('ffprobe') or custom_path.replace('ffmpeg', 'ffprobe')

    except Exception as e:
        result['available'] = False
        result['message'] = f"❌ FFmpegの確認に失敗しました: {e}\nパス: {custom_path}"

    return result


def setup_ffmpeg_for_pydub():
    """PyDub用にFFmpegを設定"""
    # 既に設定されている場合はスキップ
    if hasattr(AudioSegment, 'ffmpeg') and AudioSegment.ffmpeg:
        return True

    # システムのPATHからFFmpegを検索
    ffmpeg_path = shutil.which('ffmpeg')

    if ffmpeg_path and os.path.exists(ffmpeg_path):
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        AudioSegment.ffprobe = shutil.which('ffprobe') or ffmpeg_path.replace('ffmpeg', 'ffprobe')
        return True

    # imageio-ffmpegを試す
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_path):
            AudioSegment.converter = ffmpeg_path
            AudioSegment.ffmpeg = ffmpeg_path
            return True
    except ImportError:
        pass

    return False


class AudioProcessorGUI:
    """音声処理のGUIラッパークラス"""

    def __init__(self):
        self.temp_files = []
        # PyDub用のFFmpeg設定を初期化
        setup_ffmpeg_for_pydub()

    def log(self, message: str):
        """ログメッセージを出力"""
        print(f"[LOG] {message}")

    def cleanup_temp_files(self):
        """一時ファイルをクリーンアップ"""
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        self.temp_files = []

    def extract_audio_from_video(self, video_path: str, output_path: str, progress=gr.Progress()) -> bool:
        """MP4から音声を抽出"""
        try:
            progress(0.1, desc="動画ファイルを読み込み中...")
            video = VideoFileClip(video_path)

            if video.audio is None:
                video.close()
                return False, "エラー: 動画ファイルに音声トラックが含まれていません"

            progress(0.3, desc="音声を抽出中...")
            video.audio.write_audiofile(
                output_path,
                codec='pcm_s16le',
                verbose=False,
                logger=None
            )

            video.close()
            self.temp_files.append(output_path)
            progress(0.5, desc="音声抽出完了")
            return True, "音声抽出完了"

        except Exception as e:
            error_msg = f"エラー: 音声抽出に失敗しました: {e}"
            if "ffmpeg" in str(e).lower():
                error_msg += "\nFFmpegがインストールされていない可能性があります。"
            return False, error_msg

    def reduce_noise(self, input_path: str, output_path: str, progress=gr.Progress()) -> bool:
        """ノイズ除去を実行"""
        try:
            progress(0.5, desc="音声ファイルを読み込み中...")
            audio = AudioSegment.from_file(input_path)

            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))

            sample_rate = audio.frame_rate

            # 音声の長さをチェック
            audio_length = len(samples) / sample_rate
            self.log(f"音声の長さ: {audio_length:.2f}秒, サンプルレート: {sample_rate}Hz")

            # 音声が短すぎる場合はスキップ
            if audio_length < 0.5:
                self.log("音声が短すぎるため、ノイズ除去をスキップします")
                audio.export(output_path, format="wav")
                self.temp_files.append(output_path)
                return True, "ノイズ除去スキップ（音声が短いため）"

            progress(0.6, desc="ノイズ除去処理を実行中...")

            # noisereduceのデフォルトパラメータを使用
            # n_fftやhop_lengthを手動で指定するとSTFTパラメータエラーが発生する場合があるため
            self.log(f"ノイズ除去を実行中（デフォルトパラメータ使用）...")

            try:
                # ステレオの場合、各チャンネルを個別に処理
                if audio.channels == 2:
                    self.log("ステレオ音声: 各チャンネルを個別に処理します")
                    # 左チャンネル
                    left_channel = samples[:, 0]
                    reduced_left = nr.reduce_noise(
                        y=left_channel,
                        sr=sample_rate,
                        stationary=True,
                        prop_decrease=0.8
                    )

                    # 右チャンネル
                    right_channel = samples[:, 1]
                    reduced_right = nr.reduce_noise(
                        y=right_channel,
                        sr=sample_rate,
                        stationary=True,
                        prop_decrease=0.8
                    )

                    # 2チャンネルを結合
                    reduced_noise = np.column_stack((reduced_left, reduced_right))
                else:
                    # モノラル音声
                    self.log("モノラル音声を処理します")
                    reduced_noise = nr.reduce_noise(
                        y=samples,
                        sr=sample_rate,
                        stationary=True,
                        prop_decrease=0.8
                    )
            except Exception as nr_error:
                # ノイズ除去に失敗した場合は元の音声を使用
                self.log(f"ノイズ除去エラー: {nr_error}、元の音声を使用します")
                audio.export(output_path, format="wav")
                self.temp_files.append(output_path)
                return True, "ノイズ除去スキップ（処理エラーのため元の音声を使用）"

            # ステレオの場合は2チャンネルをインターリーブ形式にフラット化
            if audio.channels == 2:
                # column_stackで結合したので (N, 2) の形状になっている
                # AudioSegmentに渡すために (N*2,) の1次元配列にする
                reduced_noise = reduced_noise.flatten()

            reduced_noise = reduced_noise.astype(np.int16)

            processed_audio = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )

            progress(0.7, desc="ノイズ除去済み音声を保存中...")
            processed_audio.export(output_path, format="wav")
            self.temp_files.append(output_path)
            return True, "ノイズ除去完了"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] ノイズ除去エラー: {error_details}")
            return False, f"エラー: ノイズ除去に失敗しました: {e}"

    def normalize_audio(self, input_path: str, output_path: str, target_dBFS: float, progress=gr.Progress()) -> bool:
        """音量正規化を実行"""
        try:
            progress(0.7, desc="音声ファイルを読み込み中...")
            audio = AudioSegment.from_file(input_path)

            progress(0.75, desc=f"音量を正規化中 (目標: {target_dBFS} dBFS)...")
            normalized = normalize(audio)
            change_in_dBFS = target_dBFS - normalized.dBFS
            normalized = normalized.apply_gain(change_in_dBFS)

            progress(0.8, desc="正規化済み音声を保存中...")
            normalized.export(output_path, format="wav")
            self.temp_files.append(output_path)
            return True, "音量正規化完了"

        except Exception as e:
            return False, f"エラー: 音量正規化に失敗しました: {e}"

    def apply_compression(self, input_path: str, output_path: str, progress=gr.Progress()) -> bool:
        """ダイナミックレンジ圧縮を適用"""
        try:
            progress(0.8, desc="音声ファイルを読み込み中...")
            audio = AudioSegment.from_file(input_path)

            progress(0.85, desc="ダイナミックレンジ圧縮を適用中...")
            compressed = compress_dynamic_range(
                audio,
                threshold=-20.0,
                ratio=4.0,
                attack=5.0,
                release=50.0
            )

            progress(0.9, desc="圧縮済み音声を保存中...")
            compressed.export(output_path, format="wav")
            self.temp_files.append(output_path)
            return True, "ダイナミックレンジ圧縮完了"

        except Exception as e:
            return False, f"エラー: ダイナミックレンジ圧縮に失敗しました: {e}"

    def remove_silence(
        self,
        input_path: str,
        output_path: str,
        silence_thresh: int,
        min_silence_len: int,
        keep_silence: int,
        progress=gr.Progress()
    ) -> tuple:
        """無音部分を除去"""
        try:
            progress(0.9, desc="音声ファイルを読み込み中...")
            audio = AudioSegment.from_file(input_path)

            progress(0.92, desc="無音部分を検出中...")
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                seek_step=10
            )

            if not nonsilent_ranges:
                audio.export(output_path, format="wav")
                return True, "警告: 音声全体が無音として検出されました。元のファイルを使用します。", 0

            progress(0.95, desc="無音部分を除去中...")
            output_audio = AudioSegment.empty()
            for start, end in nonsilent_ranges:
                start = max(0, start - keep_silence)
                end = min(len(audio), end + keep_silence)
                output_audio += audio[start:end]

            original_duration = len(audio) / 1000.0
            new_duration = len(output_audio) / 1000.0
            removed_duration = original_duration - new_duration

            progress(0.98, desc="処理後の音声を保存中...")
            output_audio.export(output_path, format="wav")

            return True, f"無音除去完了: {removed_duration:.2f}秒の無音を削除", removed_duration

        except Exception as e:
            return False, f"エラー: 無音除去に失敗しました: {e}", 0

    def export_final_audio(
        self,
        input_path: str,
        output_path: str,
        output_format: str,
        bitrate: str,
        progress=gr.Progress()
    ) -> tuple:
        """最終音声を指定フォーマットで出力"""
        try:
            # 圧縮フォーマットの場合、FFmpegを再確認して設定
            format_lower = output_format.lower()
            if format_lower in ['mp3', 'aac', 'ogg', 'opus']:
                # FFmpegパスを取得
                ffmpeg_path = shutil.which('ffmpeg')
                if not ffmpeg_path:
                    # imageio-ffmpegを試す
                    try:
                        import imageio_ffmpeg
                        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    except ImportError:
                        pass

                if not ffmpeg_path or not os.path.exists(ffmpeg_path):
                    error_msg = f"エラー: FFmpegが見つかりません。{format_lower.upper()}形式での出力にはFFmpegが必要です。\n\n"
                    error_msg += "FFmpegのインストール方法:\n"
                    error_msg += "  Windows: https://ffmpeg.org/download.html からダウンロード\n"
                    error_msg += "  macOS: brew install ffmpeg\n"
                    error_msg += "  Linux: sudo apt install ffmpeg\n\n"
                    error_msg += "または、画面左側のFFmpeg設定セクションからカスタムパスを設定してください。\n"
                    error_msg += "代替案: 出力フォーマットをWAVに変更してください。"
                    return False, error_msg

                # PyDub用にFFmpegパスを明示的に設定（export直前）
                self.log(f"FFmpegパスを設定: {ffmpeg_path}")
                AudioSegment.converter = ffmpeg_path
                AudioSegment.ffmpeg = ffmpeg_path
                AudioSegment.ffprobe = ffmpeg_path.replace('ffmpeg', 'ffprobe')

                # 環境変数にも設定
                os.environ['FFMPEG_BINARY'] = ffmpeg_path
                os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_path

                self.log(f"AudioSegment.converter: {AudioSegment.converter}")
                self.log(f"AudioSegment.ffmpeg: {AudioSegment.ffmpeg}")

            progress(0.95, desc=f"{output_format.upper()}形式で出力中...")

            self.log(f"音声ファイルを読み込み中: {input_path}")
            audio = AudioSegment.from_file(input_path)

            self.log(f"音声を {format_lower.upper()} 形式で出力中 (ビットレート: {bitrate})...")

            # フォーマット別のパラメータ設定
            export_params = {
                'format': format_lower,
            }

            # 圧縮フォーマットの場合はビットレートを設定
            if format_lower in ['mp3', 'aac', 'ogg', 'opus']:
                export_params['bitrate'] = bitrate

                # MP3の場合はコーデックを指定
                if format_lower == 'mp3':
                    export_params['codec'] = 'libmp3lame'
                # AACの場合
                elif format_lower == 'aac':
                    export_params['codec'] = 'aac'
                # Opusの場合
                elif format_lower == 'opus':
                    export_params['codec'] = 'libopus'

            # 音声をエクスポート
            self.log(f"エクスポートパラメータ: {export_params}")
            self.log(f"出力パス: {output_path}")
            audio.export(output_path, **export_params)
            progress(1.0, desc="出力完了")
            self.log(f"音声を保存しました: {output_path}")

            return True, f"{format_lower.upper()}形式での出力完了 (ビットレート: {bitrate})"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.log(f"エラー詳細: {error_details}")

            error_msg = f"エラー: 最終音声の出力に失敗しました: {e}\n\n"
            if "codec" in str(e).lower() or "encoder" in str(e).lower() or "WinError 2" in str(e) or "FileNotFoundError" in str(e):
                error_msg += "FFmpegパスの問題の可能性があります。\n"
                error_msg += f"検出されたFFmpegパス: {shutil.which('ffmpeg') or 'なし'}\n"
                error_msg += f"AudioSegment.ffmpeg: {getattr(AudioSegment, 'ffmpeg', 'なし')}\n"
                error_msg += f"AudioSegment.converter: {getattr(AudioSegment, 'converter', 'なし')}\n\n"
                error_msg += "解決方法:\n"
                error_msg += "1. FFmpegをシステムのPATHに追加してください\n"
                error_msg += "2. または、画面左側のFFmpeg設定セクションからカスタムパスを設定\n"
                error_msg += "3. 代替案として出力フォーマットをWAVに変更"
            return False, error_msg

    def process_audio(
        self,
        input_file,
        enable_noise_reduction: bool,
        enable_silence_removal: bool,
        enable_compression: bool,
        silence_threshold: int,
        min_silence_len: int,
        keep_silence: int,
        normalize_level: float,
        output_format: str,
        bitrate: str,
        progress=gr.Progress()
    ):
        """音声処理のメイン処理"""

        # 入力ファイルの検証
        if input_file is None:
            return None, "エラー: 入力ファイルを選択してください", None

        try:
            # 古い一時ファイルをクリーンアップ
            self.cleanup_temp_files()

            progress(0, desc="処理を開始します...")

            # 入力ファイル情報を取得
            # Gradioのバージョンによって返り値が異なるため、柔軟に対応
            input_path = None

            # デバッグ情報
            print(f"[DEBUG] input_file type: {type(input_file)}")
            print(f"[DEBUG] input_file value: {input_file}")

            # 辞書型の場合（Gradio 4.x）
            if isinstance(input_file, dict):
                # 'name'キーが存在する場合
                if 'name' in input_file:
                    input_path = input_file['name']
                # 'path'キーが存在する場合
                elif 'path' in input_file:
                    input_path = input_file['path']
                else:
                    return None, f"エラー: 入力ファイルの形式が不正です（辞書型）: {input_file}", None
            # 文字列の場合
            elif isinstance(input_file, str):
                input_path = input_file
            # オブジェクトの場合
            elif hasattr(input_file, 'name'):
                input_path = input_file.name
            else:
                return None, f"エラー: 入力ファイルの形式が不正です（型: {type(input_file)}）", None

            print(f"[DEBUG] extracted path: {input_path}")

            # パスが取得できなかった場合
            if not input_path:
                return None, "エラー: ファイルパスを取得できませんでした", None

            # ファイルの存在確認
            if not os.path.exists(input_path):
                return None, f"エラー: ファイルが見つかりません: {input_path}", None

            # ディレクトリでないことを確認
            if os.path.isdir(input_path):
                return None, f"エラー: フォルダではなくファイルを選択してください: {input_path}", None

            input_size = os.path.getsize(input_path) / (1024 * 1024)
            base_name = Path(input_path).stem

            status_messages = []
            status_messages.append(f"入力ファイル: {os.path.basename(input_path)}")
            status_messages.append(f"ファイルサイズ: {input_size:.2f} MB")

            # 一時ディレクトリを作成
            temp_dir = tempfile.mkdtemp()

            # 1. 音声抽出
            status_messages.append("\n[1/7] MP4から音声を抽出中...")
            temp_audio = os.path.join(temp_dir, f"{base_name}_temp_audio.wav")
            success, msg = self.extract_audio_from_video(input_path, temp_audio, progress)
            if not success:
                self.cleanup_temp_files()
                return None, msg, None
            status_messages.append(f"✓ {msg}")
            current_file = temp_audio

            # 2. ノイズ除去
            if enable_noise_reduction:
                status_messages.append("\n[2/7] ノイズを除去中...")
                denoised_file = os.path.join(temp_dir, f"{base_name}_denoised.wav")
                success, msg = self.reduce_noise(current_file, denoised_file, progress)
                if not success:
                    self.cleanup_temp_files()
                    return None, msg, None
                status_messages.append(f"✓ {msg}")
                current_file = denoised_file
            else:
                status_messages.append("\n[2/7] ノイズ除去をスキップ")

            # 3. 音量正規化
            status_messages.append(f"\n[3/7] 音量を正規化中 (目標: {normalize_level} dBFS)...")
            normalized_file = os.path.join(temp_dir, f"{base_name}_normalized.wav")
            success, msg = self.normalize_audio(current_file, normalized_file, normalize_level, progress)
            if not success:
                self.cleanup_temp_files()
                return None, msg, None
            status_messages.append(f"✓ {msg}")
            current_file = normalized_file

            # 4. ダイナミックレンジ圧縮
            if enable_compression:
                status_messages.append("\n[4/7] ダイナミックレンジを圧縮中...")
                compressed_file = os.path.join(temp_dir, f"{base_name}_compressed.wav")
                success, msg = self.apply_compression(current_file, compressed_file, progress)
                if not success:
                    self.cleanup_temp_files()
                    return None, msg, None
                status_messages.append(f"✓ {msg}")
                current_file = compressed_file
            else:
                status_messages.append("\n[4/7] ダイナミックレンジ圧縮をスキップ")

            # 5. 無音除去
            if enable_silence_removal:
                status_messages.append("\n[5/7] 無音部分を除去中...")
                silence_removed_file = os.path.join(temp_dir, f"{base_name}_silence_removed.wav")
                success, msg, removed = self.remove_silence(
                    current_file,
                    silence_removed_file,
                    silence_threshold,
                    min_silence_len,
                    keep_silence,
                    progress
                )
                if not success:
                    self.cleanup_temp_files()
                    return None, msg, None
                status_messages.append(f"✓ {msg}")
                current_file = silence_removed_file
            else:
                status_messages.append("\n[5/7] 無音除去をスキップ")

            # 6. 最終出力（フォーマット変換）
            status_messages.append(f"\n[6/7] {output_format.upper()}形式で出力中...")
            output_file = os.path.join(temp_dir, f"{base_name}_processed.{output_format}")
            success, msg = self.export_final_audio(
                current_file,
                output_file,
                output_format,
                bitrate,
                progress
            )
            if not success:
                self.cleanup_temp_files()
                return None, msg, None
            status_messages.append(f"✓ {msg}")

            # 7. 完了
            progress(1.0, desc="処理完了！")
            status_messages.append("\n[7/7] 処理完了")

            # 出力ファイル情報
            output_size = os.path.getsize(output_file) / (1024 * 1024)

            # 音声の長さを取得
            try:
                input_audio = AudioSegment.from_file(input_path)
                output_audio = AudioSegment.from_file(output_file)
                input_duration = len(input_audio) / 1000.0
                output_duration = len(output_audio) / 1000.0
            except:
                input_duration = 0
                output_duration = 0

            # 統計情報を作成
            stats = []
            stats.append("=" * 50)
            stats.append("処理結果")
            stats.append("=" * 50)
            stats.append(f"\n出力ファイルサイズ: {output_size:.2f} MB")

            if input_duration > 0 and output_duration > 0:
                stats.append(f"元の音声時間: {input_duration:.2f}秒 ({input_duration/60:.2f}分)")
                stats.append(f"処理後の音声時間: {output_duration:.2f}秒 ({output_duration/60:.2f}分)")
                time_saved = input_duration - output_duration
                if time_saved > 0:
                    stats.append(f"削減された時間: {time_saved:.2f}秒 ({time_saved/60:.2f}分)")

            if input_size > 0 and output_size > 0:
                size_ratio = (output_size / input_size) * 100
                stats.append(f"\nファイルサイズ変化: {input_size:.2f} MB → {output_size:.2f} MB")
                stats.append(f"圧縮率: {size_ratio:.1f}%")

            stats.append("\n" + "=" * 50)

            # ステータスメッセージと統計情報を結合
            full_status = "\n".join(status_messages) + "\n\n" + "\n".join(stats)

            return output_file, full_status, output_file

        except Exception as e:
            self.cleanup_temp_files()
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] {error_details}")
            return None, f"予期しないエラーが発生しました: {e}", None


def create_gui():
    """Gradio GUIを作成"""

    processor = AudioProcessorGUI()

    # カスタムCSS
    custom_css = """
    .gradio-container {
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .output-text {
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
    }
    """

    with gr.Blocks(title="音声文字起こし用オーディオ前処理ツール", css=custom_css, theme=gr.themes.Soft()) as app:

        gr.Markdown(
            """
            # 🎵 音声文字起こし用オーディオ前処理ツール

            MP4動画ファイルから音声を抽出し、文字起こしに最適な音質に加工します。

            ### 処理内容:
            1. **音声抽出** - MP4から音声トラックを抽出
            2. **ノイズ除去** - 背景ノイズをクリーンアップ（オプション）
            3. **音量正規化** - 音量レベルを最適化
            4. **ダイナミックレンジ圧縮** - 聞き取りやすく調整
            5. **無音除去** - 長い沈黙を削除（オプション）
            6. **フォーマット変換** - MP3などの圧縮形式で出力（ファイルサイズ削減）
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 入力ファイル")
                input_file = gr.File(
                    label="MP4ファイルを選択",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"]
                )

                gr.Markdown("## ⚙️ 処理オプション")

                with gr.Accordion("基本設定", open=True):
                    enable_noise_reduction = gr.Checkbox(
                        label="ノイズ除去を有効化",
                        value=True,
                        info="背景ノイズを除去します（処理時間が増加します）"
                    )

                    enable_compression = gr.Checkbox(
                        label="ダイナミックレンジ圧縮を有効化",
                        value=False,
                        info="音量差を圧縮します（⚠️処理が非常に遅くなります）"
                    )

                    enable_silence_removal = gr.Checkbox(
                        label="無音除去を有効化",
                        value=True,
                        info="長い沈黙を削除して音声時間を短縮します"
                    )

                    normalize_level = gr.Slider(
                        minimum=-30,
                        maximum=-10,
                        value=-20,
                        step=1,
                        label="正規化レベル (dBFS)",
                        info="音量の目標レベル（推奨: -20）"
                    )

                with gr.Accordion("出力設定", open=True):
                    output_format = gr.Dropdown(
                        choices=["mp3", "aac", "wav", "ogg", "opus"],
                        value="mp3",
                        label="出力フォーマット",
                        info="MP3推奨（品質と互換性のバランス）"
                    )

                    bitrate = gr.Dropdown(
                        choices=["128k", "192k", "256k", "320k"],
                        value="192k",
                        label="ビットレート",
                        info="192kは高品質で適度なファイルサイズ（WAVには適用されません）"
                    )

                with gr.Accordion("無音除去の詳細設定", open=False):
                    silence_threshold = gr.Slider(
                        minimum=-50,
                        maximum=-25,
                        value=-40,
                        step=1,
                        label="無音判定閾値 (dBFS)",
                        info="この値以下を無音と判定（推奨: -40）"
                    )

                    min_silence_len = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        value=500,
                        step=100,
                        label="最小無音時間 (ミリ秒)",
                        info="この時間以上の無音を削除対象に（推奨: 500）"
                    )

                    keep_silence = gr.Slider(
                        minimum=0,
                        maximum=500,
                        value=100,
                        step=50,
                        label="残す無音時間 (ミリ秒)",
                        info="削除する無音の前後に残す時間（推奨: 100）"
                    )

                with gr.Accordion("🔧 FFmpeg設定", open=False):
                    ffmpeg_status = gr.Textbox(
                        label="FFmpeg状態",
                        value="確認中...",
                        lines=5,
                        interactive=False,
                        elem_classes=["output-text"]
                    )

                    ffmpeg_check_btn = gr.Button("🔄 FFmpegを再確認", size="sm")

                    ffmpeg_custom_path = gr.Textbox(
                        label="カスタムFFmpegパス（オプション）",
                        placeholder="例: C:\\ffmpeg\\bin\\ffmpeg.exe または /usr/local/bin/ffmpeg",
                        info="システムのFFmpegが見つからない場合、手動でパスを指定できます"
                    )

                    ffmpeg_set_btn = gr.Button("✅ カスタムパスを設定", size="sm")

                process_btn = gr.Button("🚀 処理を開始", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("## 📊 処理状況")
                status_output = gr.Textbox(
                    label="ステータス",
                    lines=20,
                    elem_classes=["output-text"],
                    show_copy_button=True
                )

                gr.Markdown("## 🎧 処理結果")
                audio_output = gr.Audio(
                    label="処理済み音声"
                )

                download_output = gr.File(
                    label="ダウンロード"
                )

        # プリセットボタン
        with gr.Row():
            gr.Markdown("### 🎯 クイック設定プリセット")

        with gr.Row():
            preset_standard = gr.Button("📝 標準（バランス型）")
            preset_quality = gr.Button("⭐ 高品質（ノイズ除去重視）")
            preset_fast = gr.Button("⚡ 高速（処理速度重視）")
            preset_aggressive = gr.Button("✂️ 積極的（無音削除重視）")

        # プリセット設定の関数
        def apply_standard_preset():
            return True, True, False, -40, 500, 100, -20.0

        def apply_quality_preset():
            return True, True, False, -35, 400, 150, -18.0

        def apply_fast_preset():
            return False, True, False, -40, 500, 100, -20.0

        def apply_aggressive_preset():
            return True, True, False, -45, 1000, 50, -20.0

        # プリセットボタンのイベント
        preset_standard.click(
            fn=apply_standard_preset,
            outputs=[enable_noise_reduction, enable_silence_removal, enable_compression,
                    silence_threshold, min_silence_len, keep_silence, normalize_level]
        )

        preset_quality.click(
            fn=apply_quality_preset,
            outputs=[enable_noise_reduction, enable_silence_removal, enable_compression,
                    silence_threshold, min_silence_len, keep_silence, normalize_level]
        )

        preset_fast.click(
            fn=apply_fast_preset,
            outputs=[enable_noise_reduction, enable_silence_removal, enable_compression,
                    silence_threshold, min_silence_len, keep_silence, normalize_level]
        )

        preset_aggressive.click(
            fn=apply_aggressive_preset,
            outputs=[enable_noise_reduction, enable_silence_removal, enable_compression,
                    silence_threshold, min_silence_len, keep_silence, normalize_level]
        )

        # FFmpeg確認ボタンのイベント
        def check_ffmpeg_status():
            """FFmpegの状態を確認してメッセージを返す"""
            result = check_ffmpeg_availability()
            return result['message']

        def set_custom_ffmpeg_path(path):
            """カスタムFFmpegパスを設定してメッセージを返す"""
            result = set_ffmpeg_path(path)
            return result['message']

        ffmpeg_check_btn.click(
            fn=check_ffmpeg_status,
            outputs=ffmpeg_status
        )

        ffmpeg_set_btn.click(
            fn=set_custom_ffmpeg_path,
            inputs=ffmpeg_custom_path,
            outputs=ffmpeg_status
        )

        # ページロード時にFFmpegの状態を確認
        app.load(
            fn=check_ffmpeg_status,
            outputs=ffmpeg_status
        )

        # 処理ボタンのイベント
        process_btn.click(
            fn=processor.process_audio,
            inputs=[
                input_file,
                enable_noise_reduction,
                enable_silence_removal,
                enable_compression,
                silence_threshold,
                min_silence_len,
                keep_silence,
                normalize_level,
                output_format,
                bitrate
            ],
            outputs=[audio_output, status_output, download_output]
        )

        gr.Markdown(
            """
            ---
            ### 💡 ヒント
            - **ノイズ除去**: 効果的ですが処理時間が長くなります
            - **無音閾値**: 値を小さくすると(-45など)より多くの無音を削除
            - **正規化レベル**: -20 dBFSが文字起こしサービスに最適
            - **出力フォーマット**: MP3は互換性が高く、192kビットレートで高品質
            - **ファイルサイズ削減**: MP3/AAC形式を使用すると元のMP4より小さくなります
            - **処理時間**: ファイルサイズと有効な処理により変動します

            ### ⚠️ 注意事項
            - FFmpegがシステムにインストールされている必要があります
            - 大きなファイルは処理に時間がかかる場合があります
            - ブラウザを閉じると処理が中断されます
            """
        )

    return app


def main():
    """メイン関数"""
    app = create_gui()

    print("=" * 60)
    print("音声文字起こし用オーディオ前処理ツール - Webインターフェース")
    print("=" * 60)
    print("\nブラウザでアプリケーションを開いています...")
    print("終了するには Ctrl+C を押してください")
    print("=" * 60)

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
