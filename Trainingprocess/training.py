import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from torchmetrics.detection import MeanAveragePrecision
from dataset_utils import PascalVOCDataset, get_transform, collate_fn
from model_utils import get_model_instance_segmentation

TRAIN_DATA_DIR = '/content/train'
VAL_DATA_DIR = '/content/valid'
TEST_DATA_DIR = '/content/test'

NUM_CLASSES = 4

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 0.005
NUM_EPOCHS = 15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_PATH = "faster_rcnn_vehicle_detector.pth"
BEST_MODEL_SAVE_PATH = "best_model30.pth"

def main():
    print(f"Using device: {DEVICE}")

    # 1. Load Dataset
    print("Loading datasets...")
    dataset_train = PascalVOCDataset(TRAIN_DATA_DIR, get_transform(train=True))
    dataset_val = PascalVOCDataset(VAL_DATA_DIR, get_transform(train=False))
    dataset_test = PascalVOCDataset(TEST_DATA_DIR, get_transform(train=False))

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    print(f"Train samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")
    print(f"Test samples: {len(dataset_test)}")

    # 2. Inisialisasi Model
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)

    # 3. Optimizer dan Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Inisialisasi variabel untuk menyimpan model terbaik
    best_val_loss = float('inf')

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Training")
        for images, targets in pbar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            filtered_images = []
            filtered_targets = []
            for img, tgt in zip(images, targets):
                if 'boxes' in tgt and tgt['boxes'].numel() > 0:
                    filtered_images.append(img)
                    filtered_targets.append(tgt)
                else:
                    print(f"Skipping image with no objects or missing 'boxes' in batch: {tgt.get('image_id', 'N/A')}")

            if not filtered_images:
                print("Skipping entire batch as it contains no valid objects.")
                continue

            loss_dict = model(filtered_images, filtered_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        lr_scheduler.step()
        avg_train_loss = epoch_loss / len(data_loader_train) if len(data_loader_train) > 0 else 0.0
        print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")

        # 5. Validasi
        print(f"Evaluating on validation set (Epoch {epoch+1})...")
        model.eval()
        val_losses = []

        metric_val = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=True,
            backend='pycocotools'
        )
        metric_val = metric_val.to(DEVICE)

        with torch.no_grad():
            for images_val, targets_val in data_loader_val:
                images_val = list(image.to(DEVICE) for image in images_val)
                targets_val_processed = []
                for t in targets_val:
                    processed_t = {}
                    for k, v in t.items():
                        if isinstance(v, torch.Tensor):
                            processed_t[k] = v.to(DEVICE)
                        else:
                            processed_t[k] = v
                    targets_val_processed.append(processed_t)
                targets_val = targets_val_processed

                filtered_images_val = []
                filtered_targets_val = []
                for img_val, tgt_val in zip(images_val, targets_val):
                    if 'boxes' in tgt_val and tgt_val['boxes'].numel() > 0:
                        filtered_images_val.append(img_val)
                        filtered_targets_val.append(tgt_val)

                if not filtered_images_val:
                    continue

                model.train() 
                loss_dict_val_calc = model(filtered_images_val, filtered_targets_val)
                if isinstance(loss_dict_val_calc, dict):
                    losses_val_batch = sum(loss for loss in loss_dict_val_calc.values())
                    val_losses.append(losses_val_batch.item())
                else:
                    print(f"Warning: Expected a loss dictionary during validation, but got {type(loss_dict_val_calc)}. Skipping loss calculation for this batch.")
                model.eval() 

                predictions = model(filtered_images_val)
                
                formatted_targets_val = []
                for t_val in filtered_targets_val:
                    formatted_targets_val.append({
                        "boxes": t_val["boxes"].to(DEVICE),
                        "labels": t_val["labels"].to(DEVICE)
                    })
                
                if not predictions:
                    print("Warning: No predictions generated for a validation batch. Skipping metric update for this batch.")
                    continue
                
                processed_predictions = []
                for pred in predictions:
                    if 'boxes' in pred and pred['boxes'].numel() > 0:
                        if 'scores' not in pred or pred['scores'].numel() == 0:
                            print(f"Warning: Prediction missing 'scores' or empty. Skipping this prediction for mAP calculation.")
                            continue
                        
                        processed_predictions.append({
                            "boxes": pred["boxes"].to(DEVICE),
                            "labels": pred["labels"].to(DEVICE),
                            "scores": pred["scores"].to(DEVICE)
                        })
                    else:
                        print("Warning: Prediction has no boxes. Skipping for mAP calculation.")

                if not processed_predictions:
                    print("Warning: No valid predictions for metric update in this batch. Skipping.")
                    continue

                metric_val.update(processed_predictions, formatted_targets_val)


        avg_val_loss = sum(val_losses) / len(val_losses) if len(val_losses) > 0 else 0.0
        print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}")

        val_metrics = None
        try:
            val_metrics = metric_val.compute()
            print(f"Epoch {epoch+1} Validation Metrics:")
            print(f"  mAP: {val_metrics['map'].item():.4f}")
            print(f"  mAP_50: {val_metrics['map_50'].item():.4f}")
            print(f"  mAP_75: {val_metrics['map_75'].item():.4f}")
            
            per_class_maps = val_metrics.get('map_per_class', None)
            per_class_recalls = val_metrics.get('recall_per_class', None)
            per_class_precisions = val_metrics.get('precision_per_class', None)
            class_labels_tensor = val_metrics.get('classes', None)

            if per_class_maps is not None and class_labels_tensor is not None and len(per_class_maps) > 0 :
                print("  Per-class mAP:")
                if not hasattr(dataset_train, 'class_to_id') or not dataset_train.class_to_id:
                    print("Warning: 'class_to_id' attribute not found in dataset_train or is empty. Cannot map class IDs to names.")
                    id_to_class = {i: f"Class {i}" for i in range(NUM_CLASSES)}
                else:
                    id_to_class = {v: k for k, v in dataset_train.class_to_id.items()}

                num_classes_found = min(len(per_class_maps), len(class_labels_tensor))
                for i in range(num_classes_found):
                    class_id = class_labels_tensor[i].item()
                    class_name = id_to_class.get(class_id, f"Class {class_id}")
                    class_map = per_class_maps[i].item()
                    print(f"    {class_name}: mAP={class_map:.4f}")
                    if per_class_precisions is not None and i < len(per_class_precisions):
                            print(f"      Precision: {per_class_precisions[i].item():.4f}")
                    if per_class_recalls is not None and i < len(per_class_recalls):
                            print(f"      Recall: {per_class_recalls[i].item():.4f}")
            else:
                print("  Per-class metrics not available or not in expected format. Available keys in val_metrics:", list(val_metrics.keys()))
        except Exception as e:
            print(f"Error computing or printing validation metrics: {e}")
            if val_metrics is not None and isinstance(val_metrics, dict):
                print("Note: Error occurred after computing metrics. Available keys in val_metrics:", list(val_metrics.keys()))
            else:
                print("Note: Error likely occurred during metric computation (e.g., pycocotools not found or incorrect setup), so val_metrics may not be available.")

        if avg_val_loss < best_val_loss and len(val_losses) > 0 :
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"Saved best model to {BEST_MODEL_SAVE_PATH} based on validation loss.")
        metric_val.reset()

    print("Training finished.")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

    print("Evaluating on test set...")
    model.eval()
    test_losses = []

    metric_test = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        backend='pycocotools'
    )
    metric_test = metric_test.to(DEVICE)

    with torch.no_grad():
        for images_test, targets_test in data_loader_test:
            images_test = list(image.to(DEVICE) for image in images_test)
            targets_test_processed = []
            for t in targets_test:
                processed_t = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        processed_t[k] = v.to(DEVICE)
                    else:
                        processed_t[k] = v
                targets_test_processed.append(processed_t)
            targets_test = targets_test_processed

            filtered_images_test = []
            filtered_targets_test = []
            for img_test, tgt_test in zip(images_test, targets_test):
                if 'boxes' in tgt_test and tgt_test['boxes'].numel() > 0:
                    filtered_images_test.append(img_test)
                    filtered_targets_test.append(tgt_test)

            if not filtered_images_test:
                continue

            model.train()
            loss_dict_test_calc = model(filtered_images_test, filtered_targets_test)
            if isinstance(loss_dict_test_calc, dict):
                losses_test_batch = sum(loss for loss in loss_dict_test_calc.values())
                test_losses.append(losses_test_batch.item())
            else:
                print(f"Warning: Expected a loss dictionary during test, but got {type(loss_dict_test_calc)}. Skipping loss calculation for this batch.")
            model.eval()

            predictions = model(filtered_images_test)

            formatted_targets_test = []
            for t_test in filtered_targets_test:
                    formatted_targets_test.append({
                        "boxes": t_test["boxes"].to(DEVICE),
                        "labels": t_test["labels"].to(DEVICE)
                    })
            
            if not predictions:
                print("Warning: No predictions generated for a test batch. Skipping metric update for this batch.")
                continue

            processed_predictions_test = []
            for pred in predictions:
                if 'boxes' in pred and pred['boxes'].numel() > 0:
                    if 'scores' not in pred or pred['scores'].numel() == 0:
                        print(f"Warning: Prediction missing 'scores' or empty. Skipping this prediction for mAP calculation.")
                        continue
                    
                    processed_predictions_test.append({
                        "boxes": pred["boxes"].to(DEVICE),
                        "labels": pred["labels"].to(DEVICE),
                        "scores": pred["scores"].to(DEVICE)
                    })
                else:
                    print("Warning: Prediction has no boxes. Skipping for mAP calculation.")
            
            if not processed_predictions_test:
                print("Warning: No valid predictions for metric update in this test batch. Skipping.")
                continue

            metric_test.update(processed_predictions_test, formatted_targets_test)

    avg_test_loss = sum(test_losses) / len(test_losses) if len(test_losses) > 0 else 0.0
    print(f"Final test loss: {avg_test_loss:.4f}")
    
    test_metrics = None
    try:
        test_metrics = metric_test.compute()
        print(f"Final Test Metrics:")
        print(f"  mAP: {test_metrics['map'].item():.4f}")
        print(f"  mAP_50: {test_metrics['map_50'].item():.4f}")
        print(f"  mAP_75: {test_metrics['map_75'].item():.4f}")

        per_class_maps_test = test_metrics.get('map_per_class', None)
        class_labels_tensor_test = test_metrics.get('classes', None)
        per_class_recalls_test = test_metrics.get('recall_per_class', None)
        per_class_precisions_test = test_metrics.get('precision_per_class', None)

        if per_class_maps_test is not None and class_labels_tensor_test is not None and len(per_class_maps_test) >0:
            print("  Per-class mAP:")
            if not hasattr(dataset_train, 'class_to_id') or not dataset_train.class_to_id:
                print("Warning: 'class_to_id' attribute not found in dataset_train or is empty. Cannot map class IDs to names for test metrics.")
                id_to_class = {i: f"Class {i}" for i in range(NUM_CLASSES)}
            else:
                id_to_class = {v: k for k, v in dataset_train.class_to_id.items()}

            num_classes_found_test = min(len(per_class_maps_test), len(class_labels_tensor_test))

            for i in range(num_classes_found_test):
                class_id = class_labels_tensor_test[i].item()
                class_name = id_to_class.get(class_id, f"Class {class_id}")
                class_map = per_class_maps_test[i].item()
                print(f"    {class_name}: mAP={class_map:.4f}")
                if per_class_precisions_test is not None and i < len(per_class_precisions_test):
                        print(f"      Precision: {per_class_precisions_test[i].item():.4f}")
                if per_class_recalls_test is not None and i < len(per_class_recalls_test):
                        print(f"      Recall: {per_class_recalls_test[i].item():.4f}")
        else:
            print("  Per-class metrics not available or not in expected format for test set. Available keys in test_metrics:", list(test_metrics.keys()))
    except Exception as e:
        print(f"Error computing or printing test metrics: {e}")
        if test_metrics is not None and isinstance(test_metrics, dict):
            print("Note: Error occurred after computing metrics. Available keys in test_metrics:", list(test_metrics.keys()))
        else:
            print("Note: Error likely occurred during metric computation (e.g., pycocotools not found or incorrect setup), so test_metrics may not be available.")

    print("Test evaluation finished.")

if __name__ == '__main__':
    main()