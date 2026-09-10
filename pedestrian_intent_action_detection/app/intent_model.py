import argparse

import numpy as np
import torch

from configs import cfg
from lib.modeling import make_model


CONFIG_PATH = "configs/PIE_intent_action_relation.yaml"
CHECKPOINT_PATH = "saved_models/all_relation_original_PIE.pth"


class PedestrianIntentModel:
    def __init__(
        self,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu"
    ):
        # --------------------------------------------------
        # 1. Config
        # --------------------------------------------------
        cfg.defrost()

        cfg.merge_from_file(
            config_path
        )

        cfg.MODEL.TASK = "action_intent_single"
        cfg.MODEL.WITH_TRAFFIC = True
        cfg.DATASET.BALANCE = False

        cfg.freeze()

        self.cfg = cfg

        # --------------------------------------------------
        # 2. Device
        #
        # Legacy PyTorch 1.4 + CUDA 10.1 환경에서는
        # 최신 GPU로 model.to("cuda") 시 stall이 발생했으므로
        # 현재는 CPU inference 사용
        # --------------------------------------------------
        if (
            device == "cuda"
            and torch.cuda.is_available()
        ):
            self.device = torch.device(
                "cuda"
            )
        else:
            self.device = torch.device(
                "cpu"
            )

        print(
            f"Device: {self.device}"
        )

        # --------------------------------------------------
        # 3. Model 생성
        # --------------------------------------------------
        print(
            "[1] Creating model..."
        )

        self.model = make_model(
            self.cfg
        )

        print(
            "[2] Model created"
        )

        # --------------------------------------------------
        # 4. Checkpoint load
        #
        # checkpoint는 CPU에서 먼저 로드
        # --------------------------------------------------
        print(
            "[3] Loading checkpoint on CPU..."
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        print(
            "[4] Checkpoint loaded from disk"
        )

        # --------------------------------------------------
        # 5. Checkpoint structure
        # --------------------------------------------------
        print(
            "[5] Parsing checkpoint..."
        )

        if isinstance(
            checkpoint,
            dict
        ):
            if "model" in checkpoint:
                state_dict = (
                    checkpoint["model"]
                )

                print(
                    "Checkpoint format: "
                    "{'model': state_dict}"
                )

            elif "state_dict" in checkpoint:
                state_dict = (
                    checkpoint["state_dict"]
                )

                print(
                    "Checkpoint format: "
                    "{'state_dict': state_dict}"
                )

            else:
                state_dict = checkpoint

                print(
                    "Checkpoint format: "
                    "raw state_dict"
                )

        else:
            state_dict = checkpoint

            print(
                "Checkpoint format: "
                "non-dict object"
            )

        # --------------------------------------------------
        # 6. Weight load
        # --------------------------------------------------
        print(
            "[6] Loading state_dict..."
        )

        self.model.load_state_dict(
            state_dict
        )

        print(
            "[7] state_dict loaded"
        )

        # --------------------------------------------------
        # 7. Device
        # --------------------------------------------------
        print(
            f"[8] Moving model to "
            f"{self.device}..."
        )

        self.model.to(
            self.device
        )

        print(
            "[9] Model moved to device"
        )

        # --------------------------------------------------
        # 8. Evaluation mode
        # --------------------------------------------------
        self.model.eval()

        print(
            "Checkpoint loaded successfully: "
            f"{checkpoint_path}"
        )

    # ------------------------------------------------------
    # Empty traffic object
    # ------------------------------------------------------
    def _make_empty_traffic(
        self,
        batch_size,
        seq_len,
        feature_dim
    ):
        traffic = []

        for _ in range(
            batch_size
        ):
            traffic.append(
                torch.empty(
                    0,
                    seq_len,
                    feature_dim,
                    dtype=torch.float32,
                    device=self.device
                )
            )

        return traffic

    # ------------------------------------------------------
    # Empty class information
    # ------------------------------------------------------
    def _make_empty_class(
        self,
        batch_size,
        seq_len
    ):
        classes = []

        for _ in range(
            batch_size
        ):
            classes.append(
                torch.empty(
                    0,
                    seq_len,
                    dtype=torch.float32,
                    device=self.device
                )
            )

        return classes

    # ------------------------------------------------------
    # Zero / empty traffic dictionary
    # ------------------------------------------------------
    def _make_zero_traffic(
        self,
        batch_size,
        seq_len
    ):
        traffic = {
            # Neighbor vehicle
            "x_neighbor":
                self._make_empty_traffic(
                    batch_size,
                    seq_len,
                    4
                ),

            "cls_neighbor":
                self._make_empty_class(
                    batch_size,
                    seq_len
                ),

            # Traffic light
            "x_light":
                self._make_empty_traffic(
                    batch_size,
                    seq_len,
                    6
                ),

            "cls_light":
                self._make_empty_class(
                    batch_size,
                    seq_len
                ),

            # Traffic sign
            "x_sign":
                self._make_empty_traffic(
                    batch_size,
                    seq_len,
                    5
                ),

            "cls_sign":
                self._make_empty_class(
                    batch_size,
                    seq_len
                ),

            # Crosswalk
            "x_crosswalk":
                self._make_empty_traffic(
                    batch_size,
                    seq_len,
                    7
                ),

            "cls_crosswalk":
                self._make_empty_class(
                    batch_size,
                    seq_len
                ),

            # Station
            "x_station":
                self._make_empty_traffic(
                    batch_size,
                    seq_len,
                    7
                ),

            "cls_station":
                self._make_empty_class(
                    batch_size,
                    seq_len
                ),
        }

        return traffic

    def predict(
        self,
        visual_features,
        bbox_sequence
    ):
        # --------------------------------------------------
        # 1. NumPy → Tensor
        # --------------------------------------------------
        if not torch.is_tensor(
            visual_features
        ):
            visual_features = (
                torch.from_numpy(
                    visual_features
                )
            )

        if not torch.is_tensor(
            bbox_sequence
        ):
            bbox_sequence = (
                torch.from_numpy(
                    bbox_sequence
                )
            )

        # --------------------------------------------------
        # 2. dtype + device
        # --------------------------------------------------
        visual_features = (
            visual_features
            .float()
            .to(self.device)
        )

        bbox_sequence = (
            bbox_sequence
            .float()
            .to(self.device)
        )

        print()
        print(
            "===== Tensor Input ====="
        )

        print(
            "Visual tensor:",
            tuple(
                visual_features.shape
            )
        )

        print(
            "BBox tensor:",
            tuple(
                bbox_sequence.shape
            )
        )

        batch_size = (
            visual_features.shape[0]
        )

        seq_len = (
            visual_features.shape[1]
        )

        # --------------------------------------------------
        # 3. Ego
        #
        # 현재 CCTV pipeline에서는 실제 ego 정보를
        # 만들지 않았으므로 smoke test용 zero tensor
        # --------------------------------------------------
        x_ego = torch.zeros(
            batch_size,
            seq_len,
            4,
            dtype=torch.float32,
            device=self.device
        )

        # --------------------------------------------------
        # 4. Traffic
        # --------------------------------------------------
        x_traffic = (
            self._make_zero_traffic(
                batch_size,
                seq_len
            )
        )

        print(
            "Ego tensor:",
            tuple(
                x_ego.shape
            )
        )

        # --------------------------------------------------
        # 5. Forward
        # --------------------------------------------------
        print()
        print(
            "[10] Running forward..."
        )

        with torch.no_grad():
            output = self.model(
                x_visual=visual_features,
                x_bbox=bbox_sequence,
                x_ego=x_ego,
                x_traffic=x_traffic
            )

        print(
            "[11] Forward completed"
        )

        # --------------------------------------------------
        # 6. Output unpack
        #
        # output[0]:
        # current action detection
        # [B, T, 7]
        #
        # output[1]:
        # future action prediction
        # [B, T, 5, 7]
        #
        # output[2]:
        # intention
        # [B, T, 1]
        #
        # output[3]:
        # attention
        # --------------------------------------------------
        (
            action_scores,
            future_action_scores,
            intent_scores,
            attentions
        ) = output

        print()
        print(
            "===== Raw Model Output ====="
        )

        print(
            "Action:",
            tuple(
                action_scores.shape
            )
        )

        print(
            "Future action:",
            tuple(
                future_action_scores.shape
            )
        )

        print(
            "Intent:",
            tuple(
                intent_scores.shape
            )
        )

        print(
            "Attention frames:",
            len(attentions)
        )

        # --------------------------------------------------
        # 7. Last observation frame
        # --------------------------------------------------
        final_action_logits = (
            action_scores[
                :,
                -1,
                :
            ]
        )

        final_intent_logit = (
            intent_scores[
                :,
                -1,
                :
            ]
        )

        # --------------------------------------------------
        # 8. Probability
        #
        # Action:
        # 7-class classification → softmax
        #
        # Intent:
        # binary 1-logit → sigmoid
        # --------------------------------------------------
        final_action_probs = (
            torch.softmax(
                final_action_logits,
                dim=-1
            )
        )

        final_intent_prob = (
            torch.sigmoid(
                final_intent_logit
            )
        )

        # --------------------------------------------------
        # 9. CPU로 이동
        # --------------------------------------------------
        final_action_logits_cpu = (
            final_action_logits
            .detach()
            .cpu()
        )

        final_action_probs_cpu = (
            final_action_probs
            .detach()
            .cpu()
        )

        final_intent_logit_cpu = (
            final_intent_logit
            .detach()
            .cpu()
        )

        final_intent_prob_cpu = (
            final_intent_prob
            .detach()
            .cpu()
        )

        # --------------------------------------------------
        # 10. 결과 출력
        # --------------------------------------------------
        print()
        print(
            "======================================"
        )

        print(
            "        PEDESTRIAN INTENT RESULT"
        )

        print(
            "======================================"
        )

        intent_logit_value = float(
            final_intent_logit_cpu[
                0,
                0
            ]
        )

        intent_prob_value = float(
            final_intent_prob_cpu[
                0,
                0
            ]
        )

        print()
        print(
            "Crossing intention"
        )

        print(
            f"  Raw logit   : "
            f"{intent_logit_value:.6f}"
        )

        print(
            f"  Probability : "
            f"{intent_prob_value * 100:.2f}%"
        )

        # --------------------------------------------------
        # 11. Binary interpretation
        #
        # threshold는 현재 임시로 0.5 사용
        # --------------------------------------------------
        if intent_prob_value >= 0.5:
            intent_label = (
                "INTENT TO CROSS"
            )
        else:
            intent_label = (
                "NO CROSSING INTENT"
            )

        print(
            f"  Decision    : "
            f"{intent_label}"
        )

        # --------------------------------------------------
        # 12. Current action
        #
        # 아직 repository의 class-index mapping을
        # 정확히 확인하기 전이므로 index로 출력
        # --------------------------------------------------
        print()
        print(
            "Current action probabilities"
        )

        action_probs = (
            final_action_probs_cpu[
                0
            ]
            .numpy()
        )

        for action_idx, prob in enumerate(
            action_probs
        ):
            print(
                f"  action_{action_idx}: "
                f"{prob * 100:.2f}%"
            )

        # 가장 높은 action
        predicted_action = int(
            np.argmax(
                action_probs
            )
        )

        predicted_action_prob = float(
            action_probs[
                predicted_action
            ]
        )

        print()
        print(
            "Predicted current action"
        )

        print(
            f"  action_{predicted_action} "
            f"({predicted_action_prob * 100:.2f}%)"
        )

        # --------------------------------------------------
        # 13. Future action
        #
        # 현재 observation의 마지막 frame에서
        # 미래 5-step prediction 추출
        # --------------------------------------------------
        final_future_logits = (
            future_action_scores[
                0,
                -1
            ]
        )

        final_future_probs = (
            torch.softmax(
                final_future_logits,
                dim=-1
            )
            .detach()
            .cpu()
            .numpy()
        )

        print()
        print(
            "Future action prediction"
        )

        for future_step in range(
            final_future_probs.shape[0]
        ):
            probs = (
                final_future_probs[
                    future_step
                ]
            )

            action_idx = int(
                np.argmax(
                    probs
                )
            )

            probability = float(
                probs[
                    action_idx
                ]
            )

            print(
                f"  t+{future_step + 1}: "
                f"action_{action_idx} "
                f"({probability * 100:.2f}%)"
            )

        print(
            "======================================"
        )

        # --------------------------------------------------
        # 14. Return
        #
        # 이후 main pipeline과 연결할 때 사용할 값
        # --------------------------------------------------
        result = {
            "intent_logit":
                intent_logit_value,

            "intent_probability":
                intent_prob_value,

            "intent_label":
                intent_label,

            "action_probabilities":
                action_probs,

            "predicted_action":
                predicted_action,

            "predicted_action_probability":
                predicted_action_prob,

            "future_action_probabilities":
                final_future_probs,

            "raw_output":
                output
        }

        return result


def main():
    # --------------------------------------------------
    # 1. Arguments
    # --------------------------------------------------
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Modern pipeline에서 생성한 "
            ".npz feature 파일"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 2. NPZ
    # --------------------------------------------------
    print(
        f"Loading input: "
        f"{args.input}"
    )

    data = np.load(
        args.input,
        allow_pickle=False
    )

    # --------------------------------------------------
    # 3. Metadata
    # --------------------------------------------------
    track_id = int(
        data["track_id"]
    )

    frame_idx = int(
        data["frame_idx"]
    )

    # --------------------------------------------------
    # 4. Input tensor
    # --------------------------------------------------
    visual_features = (
        torch.from_numpy(
            data[
                "visual_features"
            ]
        )
    )

    bbox_sequence = (
        torch.from_numpy(
            data[
                "bbox_sequence"
            ]
        )
    )

    # --------------------------------------------------
    # 5. Input 확인
    # --------------------------------------------------
    print()
    print(
        "===== Input ====="
    )

    print(
        "Track ID:",
        track_id
    )

    print(
        "Frame:",
        frame_idx
    )

    print(
        "Visual:",
        tuple(
            visual_features.shape
        )
    )

    print(
        "BBox:",
        tuple(
            bbox_sequence.shape
        )
    )

    print(
        "Visual dtype:",
        visual_features.dtype
    )

    print(
        "BBox dtype:",
        bbox_sequence.dtype
    )

    print()

    # --------------------------------------------------
    # 6. Model
    # --------------------------------------------------
    intent_model = (
        PedestrianIntentModel(
            device="cpu"
        )
    )

    # --------------------------------------------------
    # 7. Inference
    # --------------------------------------------------
    print()
    print(
        "===== Inference ====="
    )

    result = (
        intent_model.predict(
            visual_features,
            bbox_sequence
        )
    )

    # --------------------------------------------------
    # 8. 최종 metadata 포함 출력
    # --------------------------------------------------
    print()
    print(
        "===== Final Summary ====="
    )

    print(
        f"Person ID : {track_id}"
    )

    print(
        f"Frame     : {frame_idx}"
    )

    print(
        "Intent    : "
        f"{result['intent_probability'] * 100:.2f}%"
    )

    print(
        "Decision  : "
        f"{result['intent_label']}"
    )

    print(
        "Action    : "
        f"action_{result['predicted_action']}"
    )


if __name__ == "__main__":
    main()