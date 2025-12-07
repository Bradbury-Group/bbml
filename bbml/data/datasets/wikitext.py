from pathlib import Path
from typing import Literal

from datasets import load_dataset, Dataset as HFDataset, load_from_disk
from torch.utils.data import Dataset

from bbml.core.constants import BASE_CACHE_PATH
from bbml.debug import fprint


class WikiTextDataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "validation", "test"] = "train",
        cache_path: str | Path = BASE_CACHE_PATH / "datasets/wikitext",
    ):
        cache_path = Path(cache_path)
        self.cache_dir = cache_path / split

        if self.cache_dir.exists():
            self.ds = load_from_disk(str(self.cache_dir))
        else:
            self.ds = self.process_and_cache(split, self.cache_dir)

    @staticmethod
    def _is_page_header(text: str) -> bool:
        """
        A line is a page header if:
        - it starts with " ="
        - and ends with "= \n"
        """
        return text.startswith(" =") and text.endswith("= \n")

    def process_and_cache(self, split: str, cache_dir: Path) -> HFDataset:
        wikitext_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        wikitext_clean = wikitext_raw.filter(
            lambda d: d["text"].strip() != ""
        )

        # simple combining (small ds)
        pages: list[str] = []
        current_page_lines: list[str] = []
        for text in wikitext_clean["text"]:
            if self._is_page_header(text) and current_page_lines: # new page
                pages.append("".join(current_page_lines))
                current_page_lines = []
            current_page_lines.append(text)

        if current_page_lines:
            pages.append("".join(current_page_lines))

        wikitext_pages = HFDataset.from_dict({"text": pages})
        cache_dir.mkdir(parents=True, exist_ok=True)
        wikitext_pages.save_to_disk(str(cache_dir))

        return wikitext_pages

    def __len__(self) -> int:
        return len(self.ds)

    @fprint
    def __getitem__(self, index: int) -> dict:
        return self.ds[index]