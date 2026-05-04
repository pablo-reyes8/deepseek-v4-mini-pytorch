import torch 

def normalize_lm_batch(batch):
    """
    Convierte el batch a formato compatible con model(**batch).
    """

    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError(
                f"El batch dict debe tener 'input_ids'. Keys disponibles: {list(batch.keys())}"
            )

        if "labels" not in batch:
            batch = dict(batch)
            batch["labels"] = batch["input_ids"]

        return batch

    if torch.is_tensor(batch):
        return {
            "input_ids": batch,
            "labels": batch,
        }

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            input_ids, labels = batch
            return {
                "input_ids": input_ids,
                "labels": labels,
            }

        if len(batch) == 3:
            input_ids, labels, attention_mask = batch
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        raise ValueError(
            f"Batch list/tuple no soportado. Esperaba longitud 2 o 3, pero llegó {len(batch)}."
        )

    raise TypeError(f"Tipo de batch no soportado: {type(batch)}")