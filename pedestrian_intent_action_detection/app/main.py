import os
import numpy as np

import cv2

from tracker import Tracker
from sequence_buffer import SequenceBuffer
from feature_extractor import PedestrianFeatureExtractor


VIDEO_PATH = "videos/test.mp4"

OUTPUT_PATH = (
    "outputs/visualization/"
    "tracking_buffer_output.mp4"
)

FEATURE_OUTPUT_DIR = "outputs/features"


def main():
    # --------------------------------------------------
    # 1. Output directory
    # --------------------------------------------------
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    os.makedirs(
        FEATURE_OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # 2. Tracker
    # YOLO + ByteTrack
    # --------------------------------------------------
    tracker = Tracker()

    # --------------------------------------------------
    # 3. Pedestrian sequence buffer
    #
    # pretrained intention model:
    # INPUT_LEN = 15
    # FPS = 30
    #
    # 약 0.5초의 보행자 sequence 사용
    # --------------------------------------------------
    sequence_buffer = SequenceBuffer(
        maxlen=15
    )

    # --------------------------------------------------
    # 4. Visual feature extractor
    # --------------------------------------------------
    feature_extractor = PedestrianFeatureExtractor(
        device="cuda"
    )

    # --------------------------------------------------
    # 5. Video open
    # --------------------------------------------------
    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"영상 파일을 열 수 없습니다: "
            f"{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video FPS: {fps}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"Total frames: "
        f"{total_frames}"
    )

    # --------------------------------------------------
    # 6. Output video writer
    # --------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"출력 영상을 생성할 수 없습니다: "
            f"{OUTPUT_PATH}"
        )

    frame_idx = 0

    # --------------------------------------------------
    # 7. Main loop
    # --------------------------------------------------
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------
        tracks = tracker.track(
            frame
        )

        for obj in tracks:
            track_id = obj["track_id"]
            class_name = obj["class_name"]
            bbox = obj["bbox"]
            confidence = obj["confidence"]

            x1, y1, x2, y2 = bbox

            # ------------------------------------------
            # Pedestrian
            # ------------------------------------------
            if class_name == "person":

                # 최근 15프레임 저장
                sequence_buffer.update(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    frame=frame,
                    bbox=bbox
                )

                buffer_length = len(
                    sequence_buffer.buffers[
                        track_id
                    ]
                )

                # --------------------------------------
                # 15-frame sequence 준비 완료
                # --------------------------------------
                if sequence_buffer.is_ready(
                    track_id
                ):
                    status = "READY"

                    sequence = (
                        sequence_buffer
                        .get_sequence(
                            track_id
                        )
                    )

                    # ----------------------------------
                    # Visual feature + BBox 생성
                    # ----------------------------------
                    (
                        visual_features,
                        bbox_sequence
                    ) = (
                        feature_extractor
                        .extract_sequence(
                            sequence
                        )
                    )

                    # ----------------------------------
                    # Feature 생성 성공
                    # ----------------------------------
                    if (
                        visual_features is not None
                        and
                        bbox_sequence is not None
                    ):
                        # 너무 많은 출력 방지
                        if frame_idx % 100 == 0:
                            print(
                                f"[Frame {frame_idx}] "
                                f"Person #{track_id} | "
                                f"Visual: "
                                f"{tuple(visual_features.shape)} "
                                f"| "
                                f"BBox: "
                                f"{tuple(bbox_sequence.shape)}"
                            )

                        # ------------------------------
                        # Modern PyTorch
                        #       ↓
                        # NumPy NPZ
                        #       ↓
                        # Legacy Docker / PyTorch 1.4
                        # ------------------------------
                        save_path = os.path.join(
                            FEATURE_OUTPUT_DIR,
                            (
                                f"person_"
                                f"{track_id}_"
                                f"latest.npz"
                            )
                        )

                        np.savez_compressed(
                            save_path,

                            track_id=np.array(
                                track_id
                            ),

                            frame_idx=np.array(
                                frame_idx
                            ),

                            visual_features=(
                                visual_features
                                .detach()
                                .cpu()
                                .numpy()
                            ),

                            bbox_sequence=(
                                bbox_sequence
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        )

                else:
                    status = (
                        f"{buffer_length}/15"
                    )

                label = (
                    f"person "
                    f"#{track_id} "
                    f"{confidence:.2f} "
                    f"[{status}]"
                )

            # ------------------------------------------
            # Vehicle
            # ------------------------------------------
            else:
                label = (
                    f"{class_name} "
                    f"#{track_id} "
                    f"{confidence:.2f}"
                )

            # ------------------------------------------
            # Bounding Box
            # ------------------------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------
        # Output frame
        # ----------------------------------------------
        writer.write(
            frame
        )

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(
                f"Processed "
                f"{frame_idx} / "
                f"{total_frames} frames"
            )

    # --------------------------------------------------
    # 8. Cleanup
    # --------------------------------------------------
    cap.release()
    writer.release()

    print()
    print(
        "Processing completed"
    )

    print(
        f"Output video: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Feature directory: "
        f"{FEATURE_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()