import runpy
from pathlib import Path
import sys
from typing import Any
import unittest

import numpy as np

import bt
import bt.nn.functional as F
from tests.utils import to_numpy


class ParameterTests(unittest.TestCase):
    def test_parameter_from_arraylike_returns_grad_tracking_float_tensor(self) -> None:
        parameter = bt.nn.Parameter([1, 2, 3])

        self.assertIsInstance(parameter, bt.Tensor)
        self.assertEqual(parameter.dtype, bt.float32)
        self.assertTrue(parameter.requires_grad)
        np.testing.assert_allclose(
            to_numpy(parameter),
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_parameter_from_tensor_detaches_from_prior_graph(self) -> None:
        source = bt.tensor(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
        intermediate = source * 2.0

        parameter = bt.nn.Parameter(intermediate)
        loss = parameter.sum()
        loss.backward()

        self.assertTrue(parameter.requires_grad)
        self.assertTrue(parameter.is_leaf)
        self.assertIsNone(source.grad)
        grad = parameter.grad
        assert grad is not None
        np.testing.assert_allclose(
            to_numpy(grad),
            np.ones((3,), dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


class ModuleTests(unittest.TestCase):
    class Leaf(bt.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.weight = bt.nn.Parameter([value])

        def forward(self, input: bt.Tensor) -> bt.Tensor:
            return input + self.weight

    class Root(bt.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = bt.nn.Parameter([1.0])
            self.child = ModuleTests.Leaf(2.0)
            self.second = bt.nn.Parameter([3.0])
            self.non_parameter = bt.tensor([4.0])

        def forward(self, input: bt.Tensor) -> bt.Tensor:
            return self.child(input) + self.first + self.second

    def test_module_registers_parameters_and_children(self) -> None:
        module = self.Root()

        parameters = tuple(module.parameters())

        self.assertEqual(parameters, (module.first, module.second, module.child.weight))

    def test_module_reassignment_updates_registration(self) -> None:
        module = self.Root()

        setattr(module, "first", None)
        setattr(module, "child", 1)
        setattr(module, "third", bt.nn.Parameter([5.0]))
        setattr(module, "plain", bt.tensor([6.0], requires_grad=False))

        parameters = tuple(module.parameters())

        self.assertEqual(parameters, (module.second, getattr(module, "third")))

    def test_train_and_eval_propagate_recursively(self) -> None:
        module = self.Root()

        self.assertTrue(module.training)
        self.assertTrue(module.child.training)

        self.assertIs(module.eval(), module)
        self.assertFalse(module.training)
        self.assertFalse(module.child.training)

        self.assertIs(module.train(), module)
        self.assertTrue(module.training)
        self.assertTrue(module.child.training)


class LinearTests(unittest.TestCase):
    def test_linear_forward_matches_manual_affine_transform(self) -> None:
        layer = bt.nn.Linear(3, 2)
        layer.weight = bt.nn.Parameter(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        layer.bias = bt.nn.Parameter(np.asarray([0.5, -1.5], dtype=np.float32))
        input_tensor = bt.tensor(np.asarray([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]], dtype=np.float32))

        out = layer(input_tensor)

        expected = np.asarray([[-1.5, -3.5], [20.5, 45.5]], dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)

    def test_linear_without_bias_matches_matmul(self) -> None:
        layer = bt.nn.Linear(2, 3, bias=False)
        layer.weight = bt.nn.Parameter(
            np.asarray([[1.0, 2.0], [0.5, -1.0], [-3.0, 4.0]], dtype=np.float32)
        )
        input_tensor = bt.tensor(np.asarray([[2.0, 1.0]], dtype=np.float32))

        out = layer(input_tensor)

        expected = np.asarray([[4.0, 0.0, -2.0]], dtype=np.float32)
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)


class EmbeddingModuleTests(unittest.TestCase):
    def test_embedding_module_registers_weight_and_matches_functional(self) -> None:
        layer = bt.nn.Embedding(5, 3)
        weight = np.asarray(np.arange(15, dtype=np.float32).reshape(5, 3), dtype=np.float32)
        layer.weight = bt.nn.Parameter(weight)
        indices = bt.tensor(np.asarray([[0, 2], [4, 1]], dtype=np.int64))

        out = layer(indices)

        self.assertEqual(tuple(layer.parameters()), (layer.weight,))
        expected = weight[np.asarray([[0, 2], [4, 1]], dtype=np.int64)]
        np.testing.assert_allclose(to_numpy(out), expected, rtol=1e-6, atol=1e-6)


class LayerNormModuleTests(unittest.TestCase):
    def test_layer_norm_module_with_affine_matches_functional(self) -> None:
        source = np.asarray(np.arange(2 * 4, dtype=np.float32).reshape(2, 4), dtype=np.float32)
        layer = bt.nn.LayerNorm(4)
        layer.weight = bt.nn.Parameter(np.asarray([1.0, 1.5, 0.5, 2.0], dtype=np.float32))
        layer.bias = bt.nn.Parameter(np.asarray([-1.0, 0.5, 2.0, -0.5], dtype=np.float32))
        input_tensor = bt.tensor(source)

        out = layer(input_tensor)
        expected = F.layer_norm(input_tensor, normalized_shape=(4,), weight=layer.weight, bias=layer.bias)

        self.assertEqual(layer.normalized_shape, (4,))
        np.testing.assert_allclose(to_numpy(out), to_numpy(expected), rtol=1e-5, atol=1e-6)

    def test_layer_norm_module_without_affine_has_no_parameters(self) -> None:
        source = np.asarray(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3), dtype=np.float32)
        layer = bt.nn.LayerNorm((2, 3), elementwise_affine=False)

        out = layer(bt.tensor(source))
        expected = F.layer_norm(bt.tensor(source), normalized_shape=(2, 3))

        self.assertEqual(tuple(layer.parameters()), ())
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)
        np.testing.assert_allclose(to_numpy(out), to_numpy(expected), rtol=1e-5, atol=1e-6)


class Milestone010ExperimentTests(unittest.TestCase):
    def _load_experiment_globals(self, script_name: str) -> dict[str, Any]:
        experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
        script_path = experiments_dir / script_name
        sys.path.insert(0, str(experiments_dir))
        try:
            return runpy.run_path(str(script_path))
        finally:
            sys.path.pop(0)

    def test_single_head_attention_experiment_smoke(self) -> None:
        module_globals = self._load_experiment_globals("010_single_head_attention_bt.py")
        model_cls = module_globals["SingleHeadAttentionLanguageModel"]
        loss_fn = module_globals["loss_fn"]
        context_length = module_globals["CONTEXT_LENGTH"]

        assert isinstance(context_length, int)
        model = model_cls(vocab_size=11)
        parameters = tuple(model.parameters())
        input_ids = bt.tensor(np.arange(2 * context_length, dtype=np.int64).reshape(2, context_length) % 11)
        target_ids = bt.tensor(
            (np.arange(2 * context_length, dtype=np.int64).reshape(2, context_length) + 1) % 11
        )

        logits = model(input_ids)
        loss = loss_fn(model, input_ids, target_ids)

        self.assertEqual(len(parameters), 8)
        self.assertEqual(logits.shape, [2, context_length, 11])
        self.assertEqual(loss.shape, [])

    def test_single_head_attention_parameter_order_and_context_validation(self) -> None:
        module_globals = self._load_experiment_globals("010_single_head_attention_bt.py")
        model_cls = module_globals["SingleHeadAttentionLanguageModel"]
        context_length = module_globals["CONTEXT_LENGTH"]

        assert isinstance(context_length, int)
        model = model_cls(vocab_size=11)

        self.assertEqual(
            tuple(model.parameters()),
            (
                model.token_embedding.weight,
                model.position_embedding.weight,
                model.query.weight,
                model.key.weight,
                model.value.weight,
                model.output.weight,
                model.lm_head.weight,
                model.lm_head.bias,
            ),
        )

        wrong_input_ids = bt.tensor(np.arange(context_length - 1, dtype=np.int64).reshape(1, -1) % 11)
        with self.assertRaisesRegex(
            ValueError,
            rf"Input sequence length {context_length - 1} does not match context length {context_length}\.",
        ):
            _ = model(wrong_input_ids)

    def test_single_head_attention_matches_raw_009_when_weights_are_copied(self) -> None:
        raw_globals = self._load_experiment_globals("009_single_head_attention_bt.py")
        module_globals = self._load_experiment_globals("010_single_head_attention_bt.py")

        init_model = raw_globals["init_model"]
        raw_forward = raw_globals["forward"]
        raw_loss_fn = raw_globals["loss_fn"]
        raw_model_params = raw_globals["model_params"]
        raw_set_seed = raw_globals["set_seed"]
        model_cls = module_globals["SingleHeadAttentionLanguageModel"]
        module_loss_fn = module_globals["loss_fn"]
        learning_rate = module_globals["LEARNING_RATE"]
        context_length = module_globals["CONTEXT_LENGTH"]

        assert isinstance(context_length, int)
        raw_set_seed(1337)
        raw_model = init_model(11)
        model = model_cls(vocab_size=11)

        model.token_embedding.weight = bt.nn.Parameter(raw_model["token_embedding_table"])
        model.position_embedding.weight = bt.nn.Parameter(raw_model["position_embedding_table"])
        model.query.weight = bt.nn.Parameter(raw_model["query_weights"].transpose(0, 1))
        model.key.weight = bt.nn.Parameter(raw_model["key_weights"].transpose(0, 1))
        model.value.weight = bt.nn.Parameter(raw_model["value_weights"].transpose(0, 1))
        model.output.weight = bt.nn.Parameter(raw_model["attention_output_weights"].transpose(0, 1))
        model.lm_head.weight = bt.nn.Parameter(raw_model["logit_weights"].transpose(0, 1))
        model.lm_head.bias = bt.nn.Parameter(raw_model["logit_bias"])

        input_ids = bt.tensor(np.arange(2 * context_length, dtype=np.int64).reshape(2, context_length) % 11)
        target_ids = bt.tensor(
            (np.arange(2 * context_length, dtype=np.int64).reshape(2, context_length) + 1) % 11
        )

        raw_logits = raw_forward(input_ids, raw_model)
        modular_logits = model(input_ids).permute([0, 2, 1])
        np.testing.assert_allclose(
            to_numpy(modular_logits),
            to_numpy(raw_logits),
            rtol=0.0,
            atol=0.0,
        )

        raw_loss = raw_loss_fn(raw_model, input_ids, target_ids)
        modular_loss = module_loss_fn(model, input_ids, target_ids)
        self.assertEqual(raw_loss.item(), modular_loss.item())

        for parameter in raw_model_params(raw_model):
            parameter.zero_grad()
        for parameter in model.parameters():
            parameter.zero_grad()

        raw_loss.backward()
        modular_loss.backward()

        raw_query_grad = raw_model["query_weights"].grad
        raw_key_grad = raw_model["key_weights"].grad
        raw_value_grad = raw_model["value_weights"].grad
        raw_output_grad = raw_model["attention_output_weights"].grad
        raw_lm_head_weight_grad = raw_model["logit_weights"].grad
        raw_token_embedding_grad = raw_model["token_embedding_table"].grad
        raw_position_embedding_grad = raw_model["position_embedding_table"].grad
        raw_lm_head_bias_grad = raw_model["logit_bias"].grad

        modular_query_grad = model.query.weight.grad
        modular_key_grad = model.key.weight.grad
        modular_value_grad = model.value.weight.grad
        modular_output_grad = model.output.weight.grad
        modular_lm_head_weight_grad = model.lm_head.weight.grad
        modular_token_embedding_grad = model.token_embedding.weight.grad
        modular_position_embedding_grad = model.position_embedding.weight.grad
        modular_lm_head_bias_grad = model.lm_head.bias.grad

        assert raw_query_grad is not None
        assert raw_key_grad is not None
        assert raw_value_grad is not None
        assert raw_output_grad is not None
        assert raw_lm_head_weight_grad is not None
        assert raw_token_embedding_grad is not None
        assert raw_position_embedding_grad is not None
        assert raw_lm_head_bias_grad is not None
        assert modular_query_grad is not None
        assert modular_key_grad is not None
        assert modular_value_grad is not None
        assert modular_output_grad is not None
        assert modular_lm_head_weight_grad is not None
        assert modular_token_embedding_grad is not None
        assert modular_position_embedding_grad is not None
        assert modular_lm_head_bias_grad is not None

        np.testing.assert_allclose(to_numpy(modular_token_embedding_grad), to_numpy(raw_token_embedding_grad))
        np.testing.assert_allclose(
            to_numpy(modular_position_embedding_grad), to_numpy(raw_position_embedding_grad)
        )
        np.testing.assert_allclose(
            to_numpy(modular_query_grad.transpose(0, 1)),
            to_numpy(raw_query_grad),
        )
        np.testing.assert_allclose(
            to_numpy(modular_key_grad.transpose(0, 1)),
            to_numpy(raw_key_grad),
        )
        np.testing.assert_allclose(
            to_numpy(modular_value_grad.transpose(0, 1)),
            to_numpy(raw_value_grad),
        )
        np.testing.assert_allclose(
            to_numpy(modular_output_grad.transpose(0, 1)),
            to_numpy(raw_output_grad),
        )
        np.testing.assert_allclose(
            to_numpy(modular_lm_head_weight_grad.transpose(0, 1)),
            to_numpy(raw_lm_head_weight_grad),
        )
        np.testing.assert_allclose(to_numpy(modular_lm_head_bias_grad), to_numpy(raw_lm_head_bias_grad))

        with bt.no_grad():
            for parameter in raw_model_params(raw_model):
                grad = parameter.grad
                assert grad is not None
                parameter -= learning_rate * grad
            for parameter in model.parameters():
                grad = parameter.grad
                assert grad is not None
                parameter -= learning_rate * grad

        np.testing.assert_allclose(
            to_numpy(model.token_embedding.weight),
            to_numpy(raw_model["token_embedding_table"]),
        )
        np.testing.assert_allclose(
            to_numpy(model.position_embedding.weight),
            to_numpy(raw_model["position_embedding_table"]),
        )
        np.testing.assert_allclose(
            to_numpy(model.query.weight.transpose(0, 1)),
            to_numpy(raw_model["query_weights"]),
        )
        np.testing.assert_allclose(
            to_numpy(model.key.weight.transpose(0, 1)),
            to_numpy(raw_model["key_weights"]),
        )
        np.testing.assert_allclose(
            to_numpy(model.value.weight.transpose(0, 1)),
            to_numpy(raw_model["value_weights"]),
        )
        np.testing.assert_allclose(
            to_numpy(model.output.weight.transpose(0, 1)),
            to_numpy(raw_model["attention_output_weights"]),
        )
        np.testing.assert_allclose(
            to_numpy(model.lm_head.weight.transpose(0, 1)),
            to_numpy(raw_model["logit_weights"]),
        )
        np.testing.assert_allclose(to_numpy(model.lm_head.bias), to_numpy(raw_model["logit_bias"]))


if __name__ == "__main__":
    unittest.main()
