from beartype import beartype
from beartype.vale import Is
import torch
from torch import Tensor
from torch.nn import Module
from os.path import isfile
import pickle
import json
from time import sleep
from functools import partial
from tqdm import tqdm
from typing import Any, Type, Annotated
from collections.abc import Callable, Iterable

@beartype
def pickle_load(filename: str) -> Any:
    with open(filename, "rb") as f:
        return pickle.load(f)
    
@beartype
def pickle_dump(filename: str, object: Any) -> None:
    with open(filename, "wb") as f:
        return pickle.dump(object, f)

@beartype
def json_load(filename: str) -> Any:
    with open(filename, "r") as f:
        return json.load(f)
    
@beartype
def json_dump(filename: str, object: Any) -> None:
    with open(filename, "w") as f:
        return json.dump(object, f)
    
@beartype
def jsonl_load(filename: str) -> list[Any]:
    with open(filename, "r") as f:
        return [json.loads(line) for line in f if line.strip() != ""]
    
@beartype
def jsonl_dump(filename: str, objects: Iterable[Any]) -> None:
    with open(filename, "w") as f:
        for object in objects:
            f.write(json.dumps(object))
            f.write("\n")
    
@beartype
def run_or_load(filename: str, function: Callable, *args, verbose_load: bool = True, **kwargs) -> Any:
    assert any(filename.endswith(format) for format in [".pickle", ".json", ".txt", ".pt"])

    if isfile(filename):
        if filename.endswith(".pickle"):
            saved_result = pickle_load(filename)
        elif filename.endswith(".json"):
            saved_result = json_load(filename)
        elif filename.endswith(".txt"):
            with open(filename, "r") as f:
                saved_result = f.read()
        elif filename.endswith(".pt"):
            saved_result = torch.load(filename)
        else:
            assert False, "unreachable"

        if verbose_load:
            print(f"Loaded from file '{filename}'.")
        return saved_result
    
    result = function(*args, **kwargs)

    if filename.endswith(".pickle"):
        pickle_dump(filename, result)
    elif filename.endswith(".json"):
        json_dump(filename, result)
    elif filename.endswith(".txt"):
        assert isinstance(result, str), \
               "Can only save a string to a txt file. " \
               "Pass a filename ending in .pickle or .json to save richer datatypes."
        with open(filename, "w") as f:
            f.write(result)
    elif filename.endswith(".pt"):
        torch.save(result, filename)
    else:
        assert False, "unreachable"

    return result

@beartype
def train_or_load( filename: str,
                   train_function: Callable,
                   saved_argument_names: list[str] = ["model"],
                   load_weights_only: bool = True,
                   **kwargs ) -> Any:

    assert all(arg_name in kwargs.keys() for arg_name in saved_argument_names)
    assert "_returned" not in saved_argument_names
    assert all(isinstance(kwargs[arg_name], Module) for arg_name in saved_argument_names)
    assert filename.endswith(".pt")

    if isfile(filename):
        saved = torch.load(filename, weights_only=load_weights_only)
        for arg_name in saved_argument_names:
            kwargs[arg_name].load_state_dict(saved[arg_name])
        print(f"Loaded from file '{filename}'.")
        return saved["_returned"]
    
    returned = train_function(**kwargs)

    saved = {arg_name: kwargs[arg_name].state_dict() for arg_name in saved_argument_names}
    saved["_returned"] = returned
    torch.save(saved, filename)

    return returned

@beartype
def NamedTensor(*names: str) -> Type:
    return Annotated[Tensor, Is[lambda tensor: tensor.names == names]]

@beartype
def multi_tqdm(num_tqdms: int, disable: bool = False) -> list[Callable]:
    if disable:
        def dummy(itr, total=None, desc=None, disable=None):
            return itr
        return [dummy] * num_tqdms
    
    def itr_wrapper(itr, progress_bar, total=None, desc=None, disable=None):
        progress_bar.n = 0
        progress_bar.desc = desc
        if total is not None:
            progress_bar.total = total
        elif hasattr(itr, "__len__"):
            progress_bar.total = len(itr)
        else:
            progress_bar.total = None
        progress_bar.reset()
        progress_bar.refresh()
        for item in itr:
            yield item
            progress_bar.update()
            progress_bar.refresh()

    progress_bars = [tqdm() for _ in range(num_tqdms)]
    return [partial(itr_wrapper, progress_bar=progress_bar) for progress_bar in progress_bars]

@beartype
def beep(n_beeps: int = 1, delay_between_beeps_seconds: float = 0.1) -> None:
    for i in range(n_beeps):
        if i != 0:
            sleep(0.1)
        print("\a")

