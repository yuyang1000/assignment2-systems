import torch
from model import BasicsTransformerLM
from optimizer import AdamW
import numpy as np
import numpy.typing as npt
from data import get_batch


DATASET: npt.NDArray[np.int_] = np.arange(1000)

def _get_model():
    return BasicsTransformerLM(
        vocab_size=1024,
        context_length=128,
        d_model=512,
        num_layers=2,
        num_heads=8,
        d_ff=2048,
        rope_theta=1000.0,
    )

def _get_optimizer(llm_model: BasicsTransformerLM):
    parameters = llm_model.parameters()
    return AdamW(parameters)


def _get_input(it) -> tuple[torch.Tensor, torch.Tensor]:
    # 需要构建一组输入，构造两个矩阵[batch、sequence、d_model]
    # 一个是输入，一个是期望的结果，然后拿着输出来计算loss
    return get_batch(dataset=DATASET,
                     batch_size=2,
                     context_length=128,
                     device="cpu")


if __name__ == '__main__':
    model = _get_model()
    opt = _get_optimizer(model)
    for iteration in range(10):
        input_data, target = _get_input(iteration + 1)
        output = model(input_data)

        # [batch * sequence, vocab_length]
        new_output = output.view(-1, 1024)
        # [batch * sequence ]
        new_target = target.view(-1)

        loss = torch.nn.functional.cross_entropy(new_output, new_target)
        print(f"iteration: {iteration}, loss: {loss}")
        loss.backward()
        opt.step()
        opt.zero_grad()






