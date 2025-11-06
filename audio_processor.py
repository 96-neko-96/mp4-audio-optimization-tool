#!/usr/bin/env python3
"""
音声文字起こし用オーディオ前処理ツール
MP4動画ファイルから音声を抽出し、文字起こしに最適な音質に加工します。
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
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


class AudioProcessor:
    """音声処理を行うメインクラス"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.temp_files = []

    def log(self, message: str):
        """詳細ログを出力"""
        if self.verbose:
            print(f"[詳細] {message}")

    def print_step(self, step_num: int, total_steps: int, description: str):
        """処理ステップを表示"""
        print(f"\n[{step_num}/{total_steps}] {description}")

    def get_file_info(self, filepath: str) -> Tuple[float, int]:
        """ファイルの情報を取得（サイズMB、時間秒）"""
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        try:
            audio = AudioSegment.from_file(filepath)
            duration_sec = len(audio) / 1000.0
            return size_mb, duration_sec
        except:
            return size_mb, 0.0

    def validate_input(self, input_file: str) -> bool:
        """入力ファイルの検証"""
        if not os.path.exists(input_file):
            print(f"エラー: 入力ファイルが見つかりません: {input_file}")
            return False

        if not input_file.lower().endswith('.mp4'):
            print(f"警告: 入力ファイルがMP4形式ではありません: {input_file}")
            print("処理を続行しますが、エラーが発生する可能性があります。")

        return True

    def extract_audio_from_video(self, video_path: str, output_path: str) -> bool:
        """MP4から音声を抽出"""
        try:
            self.log(f"動画ファイルを読み込み中: {video_path}")
            video = VideoFileClip(video_path)

            if video.audio is None:
                print("エラー: 動画ファイルに音声トラックが含まれていません")
                video.close()
                return False

            self.log(f"音声を抽出中: {output_path}")
            video.audio.write_audiofile(
                output_path,
                codec='pcm_s16le',
                verbose=self.verbose,
                logger='bar' if self.verbose else None
            )

            video.close()
            self.temp_files.append(output_path)
            return True

        except Exception as e:
            print(f"エラー: 音声抽出に失敗しました: {e}")
            if "ffmpeg" in str(e).lower():
                print("FFmpegがインストールされていない可能性があります。")
                print("FFmpegをインストールしてください: https://ffmpeg.org/download.html")
            return False

    def reduce_noise(self, input_path: str, output_path: str) -> bool:
        """ノイズ除去を実行"""
        try:
            self.log(f"音声ファイルを読み込み中: {input_path}")
            audio = AudioSegment.from_file(input_path)

            # AudioSegmentをnumpy配列に変換
            samples = np.array(audio.get_array_of_samples())

            # ステレオの場合は2チャンネルに分割
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
                return True

            self.log("ノイズ除去処理を実行中...")

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
                    # ノイズ除去を実行
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
                return True

            # ステレオの場合は2チャンネルをインターリーブ形式にフラット化
            if audio.channels == 2:
                # column_stackで結合したので (N, 2) の形状になっている
                # AudioSegmentに渡すために (N*2,) の1次元配列にする
                reduced_noise = reduced_noise.flatten()

            # int16に変換
            reduced_noise = reduced_noise.astype(np.int16)

            # AudioSegmentを作成
            processed_audio = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=sample_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )

            self.log(f"ノイズ除去済み音声を保存中: {output_path}")
            processed_audio.export(output_path, format="wav")
            self.temp_files.append(output_path)
            return True

        except Exception as e:
            import traceback
            if self.verbose:
                traceback.print_exc()
            print(f"エラー: ノイズ除去に失敗しました: {e}")
            return False

    def normalize_audio(self, input_path: str, output_path: str, target_dBFS: float = -20.0) -> bool:
        """音量正規化を実行"""
        try:
            self.log(f"音声ファイルを読み込み中: {input_path}")
            audio = AudioSegment.from_file(input_path)

            self.log(f"音量を正規化中 (目標: {target_dBFS} dBFS)...")
            # 正規化を実行
            normalized = normalize(audio)

            # 目標dBFSに調整
            change_in_dBFS = target_dBFS - normalized.dBFS
            normalized = normalized.apply_gain(change_in_dBFS)

            self.log(f"正規化済み音声を保存中: {output_path}")
            normalized.export(output_path, format="wav")
            self.temp_files.append(output_path)
            return True

        except Exception as e:
            print(f"エラー: 音量正規化に失敗しました: {e}")
            return False

    def apply_compression(self, input_path: str, output_path: str) -> bool:
        """ダイナミックレンジ圧縮を適用"""
        try:
            self.log(f"音声ファイルを読み込み中: {input_path}")
            audio = AudioSegment.from_file(input_path)

            self.log("ダイナミックレンジ圧縮を適用中...")
            # 圧縮を適用（閾値-20dB、比率4:1）
            compressed = compress_dynamic_range(
                audio,
                threshold=-20.0,
                ratio=4.0,
                attack=5.0,
                release=50.0
            )

            self.log(f"圧縮済み音声を保存中: {output_path}")
            compressed.export(output_path, format="wav")
            self.temp_files.append(output_path)
            return True

        except Exception as e:
            print(f"エラー: ダイナミックレンジ圧縮に失敗しました: {e}")
            return False

    def remove_silence(
        self,
        input_path: str,
        output_path: str,
        silence_thresh: int = -40,
        min_silence_len: int = 500,
        keep_silence: int = 100
    ) -> bool:
        """無音部分を除去"""
        try:
            self.log(f"音声ファイルを読み込み中: {input_path}")
            audio = AudioSegment.from_file(input_path)

            self.log(f"無音部分を検出中 (閾値: {silence_thresh} dBFS, 最小長: {min_silence_len} ms)...")
            # 非無音部分を検出
            nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                seek_step=10
            )

            if not nonsilent_ranges:
                print("警告: 音声全体が無音として検出されました。閾値を調整してください。")
                # 元のファイルをそのまま使用
                audio.export(output_path, format="wav")
                return True

            self.log(f"無音部分を除去中 ({len(nonsilent_ranges)} 個の音声セグメントを検出)...")
            # 非無音部分を結合（前後にkeep_silenceを残す）
            output_audio = AudioSegment.empty()
            for start, end in nonsilent_ranges:
                # 前後にkeep_silence分の余裕を持たせる
                start = max(0, start - keep_silence)
                end = min(len(audio), end + keep_silence)
                output_audio += audio[start:end]

            original_duration = len(audio) / 1000.0
            new_duration = len(output_audio) / 1000.0
            removed_duration = original_duration - new_duration

            self.log(f"無音除去完了: {removed_duration:.2f}秒の無音を削除しました")
            self.log(f"処理後の音声を保存中: {output_path}")
            output_audio.export(output_path, format="wav")

            return True

        except Exception as e:
            print(f"エラー: 無音除去に失敗しました: {e}")
            return False

    def export_final_audio(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "mp3",
        bitrate: str = "192k"
    ) -> bool:
        """最終音声を指定フォーマットで出力"""
        try:
            self.log(f"最終音声ファイルを読み込み中: {input_path}")
            audio = AudioSegment.from_file(input_path)

            # 出力フォーマットに応じた処理
            format_lower = output_format.lower()

            # 拡張子の整合性チェック
            output_ext = Path(output_path).suffix.lower().lstrip('.')
            if output_ext and output_ext != format_lower:
                self.log(f"警告: 出力ファイルの拡張子 '{output_ext}' がフォーマット '{format_lower}' と一致しません")

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
            audio.export(output_path, **export_params)
            self.log(f"音声を保存しました: {output_path}")

            return True

        except Exception as e:
            print(f"エラー: 最終音声の出力に失敗しました: {e}")
            if "codec" in str(e).lower() or "encoder" in str(e).lower():
                print(f"ヒント: {output_format} 形式のエンコードにはFFmpegが必要です")
            return False

    def cleanup_temp_files(self, keep_intermediate: bool = False):
        """一時ファイルをクリーンアップ"""
        if keep_intermediate:
            print(f"\n中間ファイルを保持しています ({len(self.temp_files)} ファイル):")
            for f in self.temp_files:
                if os.path.exists(f):
                    print(f"  - {f}")
        else:
            self.log("一時ファイルをクリーンアップ中...")
            for f in self.temp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                        self.log(f"削除: {f}")
                except Exception as e:
                    self.log(f"警告: ファイルの削除に失敗しました: {f} - {e}")

    def process(
        self,
        input_file: str,
        output_file: str,
        no_noise_reduction: bool = False,
        no_silence_removal: bool = False,
        silence_threshold: int = -40,
        min_silence_len: int = 500,
        keep_silence: int = 100,
        normalize_level: float = -20.0,
        save_intermediate: bool = False,
        output_format: str = "mp3",
        bitrate: str = "192k"
    ) -> bool:
        """音声処理のメイン処理"""

        print("=" * 60)
        print("音声文字起こし用オーディオ前処理ツール")
        print("=" * 60)

        # 入力ファイルの検証
        if not self.validate_input(input_file):
            return False

        # 入力ファイル情報を表示
        input_size, input_duration = self.get_file_info(input_file)
        print(f"\n入力ファイル: {input_file}")
        print(f"  サイズ: {input_size:.2f} MB")
        if input_duration > 0:
            print(f"  時間: {input_duration:.2f} 秒 ({input_duration/60:.2f} 分)")

        # 処理ステップ数を計算
        total_steps = 5  # 基本: 抽出、正規化、圧縮、最終出力、クリーンアップ
        if not no_noise_reduction:
            total_steps += 1
        if not no_silence_removal:
            total_steps += 1

        current_step = 0
        start_time = time.time()

        # ベースファイル名を取得
        base_name = Path(input_file).stem

        # 1. 音声抽出
        current_step += 1
        self.print_step(current_step, total_steps, "MP4から音声を抽出中...")
        step_start = time.time()

        temp_audio = f"{base_name}_temp_audio.wav"
        if not self.extract_audio_from_video(input_file, temp_audio):
            return False

        print(f"  完了 ({time.time() - step_start:.2f}秒)")
        current_file = temp_audio

        # 2. ノイズ除去
        if not no_noise_reduction:
            current_step += 1
            self.print_step(current_step, total_steps, "ノイズを除去中...")
            step_start = time.time()

            denoised_file = f"{base_name}_denoised.wav" if save_intermediate else f"{base_name}_temp_denoised.wav"
            if not self.reduce_noise(current_file, denoised_file):
                self.cleanup_temp_files(save_intermediate)
                return False

            print(f"  完了 ({time.time() - step_start:.2f}秒)")
            current_file = denoised_file
        else:
            print(f"\n[スキップ] ノイズ除去")

        # 3. 音量正規化
        current_step += 1
        self.print_step(current_step, total_steps, f"音量を正規化中 (目標: {normalize_level} dBFS)...")
        step_start = time.time()

        normalized_file = f"{base_name}_normalized.wav" if save_intermediate else f"{base_name}_temp_normalized.wav"
        if not self.normalize_audio(current_file, normalized_file, normalize_level):
            self.cleanup_temp_files(save_intermediate)
            return False

        print(f"  完了 ({time.time() - step_start:.2f}秒)")
        current_file = normalized_file

        # 4. ダイナミックレンジ圧縮
        current_step += 1
        self.print_step(current_step, total_steps, "ダイナミックレンジを圧縮中...")
        step_start = time.time()

        compressed_file = f"{base_name}_compressed.wav" if save_intermediate else f"{base_name}_temp_compressed.wav"
        if not self.apply_compression(current_file, compressed_file):
            self.cleanup_temp_files(save_intermediate)
            return False

        print(f"  完了 ({time.time() - step_start:.2f}秒)")
        current_file = compressed_file

        # 5. 無音除去
        if not no_silence_removal:
            current_step += 1
            self.print_step(current_step, total_steps, "無音部分を除去中...")
            step_start = time.time()

            silence_removed_file = f"{base_name}_silence_removed.wav" if save_intermediate else f"{base_name}_temp_silence_removed.wav"
            if not self.remove_silence(
                current_file,
                silence_removed_file,
                silence_threshold,
                min_silence_len,
                keep_silence
            ):
                self.cleanup_temp_files(save_intermediate)
                return False

            print(f"  完了 ({time.time() - step_start:.2f}秒)")
            current_file = silence_removed_file
        else:
            print(f"\n[スキップ] 無音除去")

        # 6. 最終出力（フォーマット変換）
        current_step += 1
        self.print_step(current_step, total_steps, f"{output_format.upper()}形式で出力中...")
        step_start = time.time()

        if not self.export_final_audio(current_file, output_file, output_format, bitrate):
            self.cleanup_temp_files(save_intermediate)
            return False

        print(f"  完了 ({time.time() - step_start:.2f}秒)")

        # 7. クリーンアップ
        current_step += 1
        self.print_step(current_step, total_steps, "処理を完了しています...")
        self.cleanup_temp_files(save_intermediate)

        # 結果を表示
        total_time = time.time() - start_time
        output_size, output_duration = self.get_file_info(output_file)

        print("\n" + "=" * 60)
        print("処理が完了しました!")
        print("=" * 60)
        print(f"\n出力ファイル: {output_file}")
        print(f"  サイズ: {output_size:.2f} MB")
        if output_duration > 0:
            print(f"  時間: {output_duration:.2f} 秒 ({output_duration/60:.2f} 分)")

        print(f"\n処理時間: {total_time:.2f}秒")

        if input_size > 0 and output_size > 0:
            print(f"\nファイルサイズ変化: {input_size:.2f} MB → {output_size:.2f} MB")
            size_ratio = (output_size / input_size) * 100
            print(f"  圧縮率: {size_ratio:.1f}%")

        if input_duration > 0 and output_duration > 0:
            time_saved = input_duration - output_duration
            if time_saved > 0:
                print(f"\n音声時間の削減: {time_saved:.2f}秒 ({time_saved/60:.2f}分)")

        return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="音声文字起こし用オーディオ前処理ツール - MP4から音声を抽出し、文字起こしに最適な音質に加工します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  基本的な使用方法:
    python audio_processor.py input.mp4

  出力ファイル名を指定:
    python audio_processor.py input.mp4 -o output.wav

  ノイズ除去をスキップ:
    python audio_processor.py input.mp4 --no-noise-reduction

  無音除去のパラメータを調整:
    python audio_processor.py input.mp4 --silence-threshold -35 --min-silence-len 1000

  中間ファイルを保存して詳細ログを表示:
    python audio_processor.py input.mp4 --save-intermediate -v
"""
    )

    parser.add_argument(
        "input",
        help="入力するMP4動画ファイル"
    )

    parser.add_argument(
        "-o", "--output",
        help="出力ファイル名 (デフォルト: input_processed.wav)",
        default=None
    )

    parser.add_argument(
        "--no-noise-reduction",
        action="store_true",
        help="ノイズ除去をスキップ"
    )

    parser.add_argument(
        "--no-silence-removal",
        action="store_true",
        help="無音除去をスキップ"
    )

    parser.add_argument(
        "--silence-threshold",
        type=int,
        default=-40,
        help="無音判定の閾値 (dBFS、デフォルト: -40)"
    )

    parser.add_argument(
        "--min-silence-len",
        type=int,
        default=500,
        help="無音として認識する最小時間 (ミリ秒、デフォルト: 500)"
    )

    parser.add_argument(
        "--keep-silence",
        type=int,
        default=100,
        help="無音部分を残す時間 (ミリ秒、デフォルト: 100)"
    )

    parser.add_argument(
        "--normalize-level",
        type=float,
        default=-20.0,
        help="正規化のターゲットレベル (dBFS、デフォルト: -20.0)"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="mp3",
        choices=["mp3", "aac", "wav", "ogg", "opus"],
        help="出力オーディオフォーマット (デフォルト: mp3)"
    )

    parser.add_argument(
        "--bitrate",
        type=str,
        default="192k",
        help="音声ビットレート (例: 128k, 192k, 256k, 320k) (デフォルト: 192k)"
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="中間ファイルを保存"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細なログ出力"
    )

    args = parser.parse_args()

    # 出力ファイル名を生成
    if args.output is None:
        base_name = Path(args.input).stem
        args.output = f"{base_name}_processed.{args.format}"

    # 処理を実行
    processor = AudioProcessor(verbose=args.verbose)

    try:
        success = processor.process(
            input_file=args.input,
            output_file=args.output,
            no_noise_reduction=args.no_noise_reduction,
            no_silence_removal=args.no_silence_removal,
            silence_threshold=args.silence_threshold,
            min_silence_len=args.min_silence_len,
            keep_silence=args.keep_silence,
            normalize_level=args.normalize_level,
            save_intermediate=args.save_intermediate,
            output_format=args.format,
            bitrate=args.bitrate
        )

        if success:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n処理が中断されました")
        processor.cleanup_temp_files(args.save_intermediate)
        sys.exit(130)
    except Exception as e:
        print(f"\n予期しないエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        processor.cleanup_temp_files(args.save_intermediate)
        sys.exit(1)


if __name__ == "__main__":
    main()
