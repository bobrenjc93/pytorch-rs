#!/usr/bin/env python3
"""Emit Burner EvaluationOutput JSON for torch.compile coverage parity."""

from __future__ import annotations

import argparse
import dis
import hashlib
import inspect
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
import types
import warnings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPOSITORY_ROOT / "tests" / "test_compile_corpus.py"
EVALUATION_ID = "eval_a61c0e71"
EVALUATOR_VERSION = "torch_compile_program_coverage_evaluator_v7"
EXPECTED_CORPUS_VERSION = "torch_compile_corpus_v11"
REFERENCE_PYTORCH_VERSION = "2.13.0"
EXPECTED_CATEGORY_WEIGHTS = {
    "tensor_arithmetic": 12,
    "broadcasting": 8,
    "modules_parameters_buffers": 8,
    "inference": 6,
    "training_autograd": 8,
    "python_control_flow": 8,
    "graph_breaks_fullgraph": 8,
    "dynamic_shapes_symbolics": 8,
    "mutation_aliasing_views": 8,
    "containers_pytrees": 6,
    "decompositions": 6,
    "custom_functions": 6,
    "recompilation_guards": 4,
    "dtype_device_transitions": 4,
}
REQUIRED_CATEGORIES = tuple(EXPECTED_CATEGORY_WEIGHTS)
EXPECTED_PUBLIC_GUARD_SCENARIOS = (
    "unary_shape_stride_requires_grad_guards",
    "binary_argument_metadata_guards",
    "requires_grad_branch_unary_cache",
    "bounded_limit_then_reset",
)
EXPECTED_HELD_OUT_GUARD_SCENARIOS = (
    "heldout_requires_grad_branch_binary_cache",
    "heldout_unary_rank3_metadata_mix",
    "heldout_binary_broadcast_metadata_mix",
)
EXPECTED_V11_CASE_MANIFEST = (
    {
        "name": "cpu_float32_unary_abs_neg",
        "held_out": False,
        "category": "tensor_arithmetic",
        "program": "cpu_float32_unary_abs_neg",
        "program_sha256": "ad23c54dea781dc58be910da3dba623fdc37fa2cef5629b7b7165756d3f13506",
        "make_inputs": "cpu_float32_unary_inputs",
        "make_inputs_sha256": "78d76dd56238040acfb24345f6352613a13241d7e3987242e99e714ddc9a2941",
        "inputs_sha256": "f71a81f2dd2217a12f7e7c8f0ead4762503a322baabc68e565aae6142e56c973",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_self_add",
        "held_out": False,
        "category": "tensor_arithmetic",
        "program": "cpu_float32_self_add",
        "program_sha256": "2ca99ad5852438119a18c8436fa4d7a8fc57a28e21c4b06147d6e1fe92f373b4",
        "make_inputs": "cpu_float32_self_add_inputs",
        "make_inputs_sha256": "336eef750bdfb680cdd271cb9a5a972d4a1cccc534dc7fb8a81f9f14e49a1359",
        "inputs_sha256": "828f5a6e678621ee80797ef1c2927222ebcfd565afa3ca4e8163590abaa16eb3",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_abs_neg_reordered",
        "held_out": False,
        "category": "tensor_arithmetic",
        "program": "cpu_float32_abs_neg_reordered",
        "program_sha256": "4c192f22269ec274060c3e5a49e91341bdebd0409ae3ab8454534b009adc83e9",
        "make_inputs": "cpu_float32_unary_inputs",
        "make_inputs_sha256": "78d76dd56238040acfb24345f6352613a13241d7e3987242e99e714ddc9a2941",
        "inputs_sha256": "f71a81f2dd2217a12f7e7c8f0ead4762503a322baabc68e565aae6142e56c973",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_repeated_unary_chain",
        "held_out": False,
        "category": "tensor_arithmetic",
        "program": "cpu_float32_repeated_unary_chain",
        "program_sha256": "6eb35f6d1d236837f6bbbb24386ca159920a63140d302817e9aaa85e29bc5339",
        "make_inputs": "cpu_float32_scalar_inputs",
        "make_inputs_sha256": "976bef3aa7887653dfe2a128a56afbac7686cc49720e58c2da6d51ad227a4678",
        "inputs_sha256": "012c44ae8d3f2a2cd163f8353911d9cda501ac52beac8044177f4dcd5cdee937",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_add_unary_composition",
        "held_out": False,
        "category": "tensor_arithmetic",
        "program": "cpu_float32_add_unary_composition",
        "program_sha256": "918a38ad10d3f8668dfccda5582901f5091f408a6030dac59f83fcbfb7638c6b",
        "make_inputs": "cpu_float32_empty_matrix_inputs",
        "make_inputs_sha256": "690504cd61f1aa82f9fb4da82fbfdb2b61c350b8aabc1703433ae19c79904c04",
        "inputs_sha256": "c9652bddaa7457fa633b7a37e894e7a5882f54f1fa5d61d368d0aa99f505a99a",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_inference_relu_no_grad",
        "held_out": False,
        "category": "inference",
        "program": "cpu_float32_inference_relu_no_grad",
        "program_sha256": "dcfe6e5fdaf599197e367b46fc2eef1f3abc34c4330badb43cf4004fd71635b0",
        "make_inputs": "cpu_float32_inference_relu_requires_grad_inputs",
        "make_inputs_sha256": "70287777649e7008e87cda5af1c64d05c924084be18008d664cc317686d217eb",
        "inputs_sha256": "123973c513c599113a29d690b3b40b270f103c5e87e9898a4cce6191071a9294",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": True,
    },
    {
        "name": "cpu_float32_detach_alias_view",
        "held_out": False,
        "category": "mutation_aliasing_views",
        "program": "cpu_float32_detach_alias_view",
        "program_sha256": "77897c0c0237a850252b20a4aba1eeaabd11a8ac4766b09ff7811360e060bee0",
        "make_inputs": "cpu_float32_detach_alias_view_inputs",
        "make_inputs_sha256": "25db55f61ba633450cf18a4aefe2854c8a2ee88aaa791b72ac1f05f098424c98",
        "inputs_sha256": "130efd361009c3dba976af943e4748f4662f442c8480f68e3043f5849ba2ef8f",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_float_identity_view",
        "held_out": False,
        "category": "dtype_device_transitions",
        "program": "cpu_float32_float_identity_view",
        "program_sha256": "8ba3f0f1ac4bb7d96a2509a397daddb511b83f15a7b252eb1504fc20d0f34895",
        "make_inputs": "cpu_float32_float_identity_view_inputs",
        "make_inputs_sha256": "bdd401b4335ce587640c7409f5fee7802733415eb3c5220025fa2ee2bf648bd6",
        "inputs_sha256": "924af58460b92ff3526e16d8cdbf6cd6862e3850c206e4d1dcbfb086df93e661",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_training_unary_neg_abs_add",
        "held_out": False,
        "category": "training_autograd",
        "program": "cpu_float32_training_unary_neg_abs_add",
        "program_sha256": "c24711b0dd929213cde7039c5e652d4f95b735f8b3f095b580eb71ff1c862a40",
        "make_inputs": "cpu_float32_training_unary_requires_grad_inputs",
        "make_inputs_sha256": "38ebeb4d744d9eb946ca1b9dc2dbaa095c212f63849bc4595836a831b6a1c8e1",
        "inputs_sha256": "b09579150ba5b84bebd343c7250b4eebb35db0b147302b9fbf53226533aa051a",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": True,
    },
    {
        "name": "cpu_float32_decomposition_square_scalar",
        "held_out": False,
        "category": "decompositions",
        "program": "cpu_float32_decomposition_square_scalar",
        "program_sha256": "3e8565b0871029c618c1d3500743397902777c4d1e20cc2c5e6a2bc6b789206d",
        "make_inputs": "cpu_float32_scalar_inputs",
        "make_inputs_sha256": "976bef3aa7887653dfe2a128a56afbac7686cc49720e58c2da6d51ad227a4678",
        "inputs_sha256": "012c44ae8d3f2a2cd163f8353911d9cda501ac52beac8044177f4dcd5cdee937",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_custom_function_unary",
        "held_out": False,
        "category": "custom_functions",
        "program": "cpu_float32_custom_function_unary",
        "program_sha256": "96792748da19fd7caece093cc0637aa52381ffb1530bdbba8242674a955498c7",
        "helper_sha256s": (
            {
                "name": "cpu_float32_custom_helper_unary",
                "source_sha256": "367af78a9f85d31f7b3ec1049c1b87f760fbe16cc2464859ae722e9d7bbbc1c8",
            },
        ),
        "make_inputs": "cpu_float32_unary_inputs",
        "make_inputs_sha256": "78d76dd56238040acfb24345f6352613a13241d7e3987242e99e714ddc9a2941",
        "inputs_sha256": "f71a81f2dd2217a12f7e7c8f0ead4762503a322baabc68e565aae6142e56c973",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_requires_grad_branch_unary",
        "held_out": False,
        "category": "python_control_flow",
        "program": "cpu_float32_requires_grad_branch_unary",
        "program_sha256": "d32515c11079b8b24c3d043162eed47422828c3f77f5afcb848f51ca1f593ac5",
        "make_inputs": "cpu_float32_control_flow_requires_grad_false_inputs",
        "make_inputs_sha256": "fc6e0b56689ee7e83d8762fa86e52a3362e9456416818403c85799d45320f170",
        "inputs_sha256": "45bc0581ad4e4b64a4752682357d741919fd0a4a7bc62b1ddb4b74761fb98b64",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_matrix_vector_add",
        "held_out": False,
        "category": "broadcasting",
        "program": "cpu_float32_matrix_vector_add",
        "program_sha256": "bf46de728e208d754ba46855e07edf2b4477abcdf78db8ce3a2d952e223e26d8",
        "make_inputs": "cpu_float32_matrix_vector_inputs",
        "make_inputs_sha256": "9eeb639db07e0bcd5e5cb6eb28a5c860fdce8c8413ef63134434beac71c5f54c",
        "inputs_sha256": "efde735c565edf1b5d5a841bca907124c22358cdcb27c7face8bed09dc4e78e7",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_matrix_vector_add_method",
        "held_out": False,
        "category": "broadcasting",
        "program": "cpu_float32_matrix_vector_add_method",
        "program_sha256": "948e134fbac20dda2a7ad51dee29f112739e4d6e1de1303de093736aa99fc033",
        "make_inputs": "cpu_float32_matrix_vector_requires_grad_inputs",
        "make_inputs_sha256": "b26daf5a5a8139b4088b90b730acfe49a8b6ad17586f3b62a0ac08a5a67d15bd",
        "inputs_sha256": "f00763fcf40a4156959ed9feeb4c8c167308c72ef946d8c9243fa3571ea17cad",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_tensor_scalar_add",
        "held_out": False,
        "category": "broadcasting",
        "program": "cpu_float32_tensor_scalar_add",
        "program_sha256": "4d5ed0a93b02c9ffa7b8929b02e06548591083c0e80545aa36301e4e00afe08c",
        "make_inputs": "cpu_float32_tensor_scalar_inputs",
        "make_inputs_sha256": "2879943cfca74c4beec1303a3bde83322dfa6a6df7c14118d4a0604709e89ac6",
        "inputs_sha256": "9624caa567ad51f572e066a48c0dab78d2b737fd5f58a47d47e5a3e2e79304d4",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_scalar_tensor_add",
        "held_out": False,
        "category": "broadcasting",
        "program": "cpu_float32_scalar_tensor_add",
        "program_sha256": "efb46380fac75875abc5afe0a152abce4f590985c5db42d4fc232cb2d22d8d9f",
        "make_inputs": "cpu_float32_scalar_tensor_inputs",
        "make_inputs_sha256": "c97640885d694619d8c2340c608f248037c130b005f1bb45870372ed780f0710",
        "inputs_sha256": "b5e285646fc9ea682ce87d90ab4723e40fe3243462c8e90919686fe09ad9b162",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_global_buffer_add",
        "held_out": False,
        "category": "modules_parameters_buffers",
        "program": "cpu_float32_global_buffer_add",
        "program_sha256": "c8d340ebd547ab4d06655ab823bbfbad1c28820fa09585506edab6b915677879",
        "make_inputs": "cpu_float32_global_buffer_add_inputs",
        "make_inputs_sha256": "91f8d2e1ccf96f3720f46f6c2d06830384ccf262130a47ac4aecf36b814920f1",
        "inputs_sha256": "537513ca52a53b9a87859027e1aa0b82d11fbd62d732cd56503a16045e3add20",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_tuple_list_output_pytree",
        "held_out": False,
        "category": "containers_pytrees",
        "program": "cpu_float32_tuple_list_output_pytree",
        "program_sha256": "6c3834c354197af76f9c39cfbe67fc249b06293498a2210507357f7c4bcbbb3c",
        "make_inputs": "cpu_float32_matrix_vector_requires_grad_inputs",
        "make_inputs_sha256": "b26daf5a5a8139b4088b90b730acfe49a8b6ad17586f3b62a0ac08a5a67d15bd",
        "inputs_sha256": "f00763fcf40a4156959ed9feeb4c8c167308c72ef946d8c9243fa3571ea17cad",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_recompile_guard_unary_metadata",
        "held_out": False,
        "category": "recompilation_guards",
        "program": "cpu_float32_recompile_guard_unary_metadata",
        "program_sha256": "9259670dbb4c78a328476722424c37087ba47f11698360e46e1139b1ba485ae5",
        "make_inputs": "cpu_float32_recompile_guard_unary_inputs",
        "make_inputs_sha256": "ecac84421a6c706ffb6fd206231fa5ae0f706cb9f23145ca307a30b069a93070",
        "inputs_sha256": "35312be0abb470e0f8b34dadcbacd7effb9db9c88ee858c22b0174a1963c03e1",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": 4,
    },
    {
        "name": "cpu_float32_recompile_guard_binary_metadata",
        "held_out": False,
        "category": "recompilation_guards",
        "program": "cpu_float32_recompile_guard_binary_metadata",
        "program_sha256": "932eb1bf1b91dc287d69ca4339034eabebc6f47fc5328ee60a583396a0986f2e",
        "make_inputs": "cpu_float32_recompile_guard_binary_inputs",
        "make_inputs_sha256": "78237478fbad18b34ba4e4fc1f4588d7476b9bed94060cc07a36b8cbb56fb22d",
        "inputs_sha256": "efde735c565edf1b5d5a841bca907124c22358cdcb27c7face8bed09dc4e78e7",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": 4,
    },
    {
        "name": "cpu_float32_recompile_limit_reset",
        "held_out": False,
        "category": "recompilation_guards",
        "program": "cpu_float32_recompile_limit_reset",
        "program_sha256": "c827815de7a0037a355d16e726b00f9137077a7b02519b7ebcb6043a2a898a20",
        "make_inputs": "cpu_float32_recompile_guard_unary_inputs",
        "make_inputs_sha256": "ecac84421a6c706ffb6fd206231fa5ae0f706cb9f23145ca307a30b069a93070",
        "inputs_sha256": "35312be0abb470e0f8b34dadcbacd7effb9db9c88ee858c22b0174a1963c03e1",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": 2,
    },
    {
        "name": "cpu_float32_heldout_broadcast_chain",
        "held_out": True,
        "category": "broadcasting",
        "program": "cpu_float32_heldout_broadcast_chain",
        "program_sha256": "ff446c26c4996d9a8b23d20001b59f8757e749f7ef600fcac9dc672777a7922a",
        "make_inputs": "cpu_float32_matrix_vector_inputs",
        "make_inputs_sha256": "9eeb639db07e0bcd5e5cb6eb28a5c860fdce8c8413ef63134434beac71c5f54c",
        "inputs_sha256": "efde735c565edf1b5d5a841bca907124c22358cdcb27c7face8bed09dc4e78e7",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_heldout_scalar_left_broadcast",
        "held_out": True,
        "category": "broadcasting",
        "program": "cpu_float32_heldout_scalar_left_broadcast",
        "program_sha256": "755a6c0f0ceebc4bb9ea1c2de9ce4c9fb3b9d23e47b5284b74d7d4c642ab11af",
        "make_inputs": "cpu_float32_scalar_tensor_inputs",
        "make_inputs_sha256": "c97640885d694619d8c2340c608f248037c130b005f1bb45870372ed780f0710",
        "inputs_sha256": "b5e285646fc9ea682ce87d90ab4723e40fe3243462c8e90919686fe09ad9b162",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_heldout_global_weight_unary_add",
        "held_out": True,
        "category": "modules_parameters_buffers",
        "program": "cpu_float32_heldout_global_weight_unary_add",
        "program_sha256": "c564334680e4ff91cd13c4f6e357b023f0c1ff3a557b2787fe23c975389794cf",
        "make_inputs": "cpu_float32_heldout_global_weight_unary_add_inputs",
        "make_inputs_sha256": "acd670aa69fcb9029b0d97c869541dd2f5158cd9e1832f8e95c0c0224a614a2f",
        "inputs_sha256": "3309d6bd5a238d115949a526c1b45db0a2a736d84a88de584b9ead464649a750",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_heldout_training_broadcast_neg_abs_add",
        "held_out": True,
        "category": "training_autograd",
        "program": "cpu_float32_heldout_training_broadcast_neg_abs_add",
        "program_sha256": "3dcec8f052c681ba85a6082a8547ebebbf682b0e37841ad351da434d41b70f57",
        "make_inputs": "cpu_float32_training_broadcast_requires_grad_inputs",
        "make_inputs_sha256": "a4880ef2eb26bc6f2becf62d1787b9ef4a9b72a4f850b8b4e07a076f47e32d33",
        "inputs_sha256": "ce11da368b183505a90f58cbe29494ae7e14cc8893827cf530c47eebcfe4d743",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": True,
    },
    {
        "name": "cpu_float32_heldout_inference_relu_broadcast_no_grad",
        "held_out": True,
        "category": "inference",
        "program": "cpu_float32_heldout_inference_relu_broadcast_no_grad",
        "program_sha256": "4ecfe35ed9ade48ddefd783919cdbc23af89bac9c696b1fc61c1624cdd95874e",
        "make_inputs": "cpu_float32_heldout_inference_relu_broadcast_inputs",
        "make_inputs_sha256": "a60d298fe3b0de4419920976dc2952a2aa3ac13a2cb7e0b1bf2c9db2cc61de39",
        "inputs_sha256": "c9ed9ce681adb3fc1770cca71b0ab5db1c1a0fabd9bf6fc4a8c9b831f907ddeb",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": True,
    },
    {
        "name": "cpu_float32_heldout_detach_alias_view",
        "held_out": True,
        "category": "mutation_aliasing_views",
        "program": "cpu_float32_heldout_detach_alias_view",
        "program_sha256": "79b2c71bdb0f4e56f33ba4055844dca68c4826cbad820a66c60b7b0ea6676cf2",
        "make_inputs": "cpu_float32_heldout_detach_alias_view_inputs",
        "make_inputs_sha256": "41a7df473962e8fae63eda143399044b23dcf353c19a9c1e6e8a56bf30369f51",
        "inputs_sha256": "b51bde0b06081b6eb85e8378169bedd5c419f83e8ca95eeaa9e6dc024c901d54",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_heldout_float_identity_rank3_view",
        "held_out": True,
        "category": "dtype_device_transitions",
        "program": "cpu_float32_heldout_float_identity_rank3_view",
        "program_sha256": "2045e4af858790176580ef8c1af7e6497473ad15e5a837497cb09bb73edd8894",
        "make_inputs": "cpu_float32_heldout_float_identity_rank3_view_inputs",
        "make_inputs_sha256": "51d4c762c92f169dfc952544cdd0d68a7eb92c7fbdf1a0a3b4ca0d3366c4d948",
        "inputs_sha256": "34435fe0d3c5b6aa7673bc0ad4e6f21ddac24806ed7a9d0479afd8456298067f",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_heldout_list_tuple_output_pytree",
        "held_out": True,
        "category": "containers_pytrees",
        "program": "cpu_float32_heldout_list_tuple_output_pytree",
        "program_sha256": "9ab6081406c70b563bc73d7618ce8520c79076fb7180c5ea2d4145a5921cbd42",
        "make_inputs": "cpu_float32_scalar_tensor_inputs",
        "make_inputs_sha256": "c97640885d694619d8c2340c608f248037c130b005f1bb45870372ed780f0710",
        "inputs_sha256": "b5e285646fc9ea682ce87d90ab4723e40fe3243462c8e90919686fe09ad9b162",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_heldout_decomposition_square_noncontiguous",
        "held_out": True,
        "category": "decompositions",
        "program": "cpu_float32_heldout_decomposition_square_noncontiguous",
        "program_sha256": "334478bfce216dd5c0ad6e728255b70bd913b78442af40396f7af64144804fb3",
        "make_inputs": "cpu_float32_recompile_guard_unary_stride_inputs",
        "make_inputs_sha256": "1e52c6eda609091cf2d8981ed92125a596a7cacd130403abe6741b560ad8f749",
        "inputs_sha256": "9857d132f7cd08b86cf70f048fa4b6ae34df30f8f5fee667b72c902acd1552fa",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_heldout_custom_function_binary",
        "held_out": True,
        "category": "custom_functions",
        "program": "cpu_float32_heldout_custom_function_binary",
        "program_sha256": "c08684ddd98deea174e30da02934b38e8bcfe674628a53b8ba567b84e104ad7b",
        "helper_sha256s": (
            {
                "name": "cpu_float32_heldout_custom_helper_binary",
                "source_sha256": "4eb7bd75d1edb201809e884e0104f8848b2cf72d8e17e73484e9674abc8ed6bd",
            },
        ),
        "make_inputs": "cpu_float32_matrix_vector_inputs",
        "make_inputs_sha256": "9eeb639db07e0bcd5e5cb6eb28a5c860fdce8c8413ef63134434beac71c5f54c",
        "inputs_sha256": "efde735c565edf1b5d5a841bca907124c22358cdcb27c7face8bed09dc4e78e7",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
    },
    {
        "name": "cpu_float32_heldout_requires_grad_branch_binary",
        "held_out": True,
        "category": "python_control_flow",
        "program": "cpu_float32_heldout_requires_grad_branch_binary",
        "program_sha256": "a6c8fd10376c28928b2a6f341edc2f6b7590a17257a8068a4dd66168dd7e12ce",
        "make_inputs": "cpu_float32_heldout_control_flow_requires_grad_false_inputs",
        "make_inputs_sha256": "c4d0607c9d4758aea3e21110756b869a4f4276cb65c30c1cfd885b06410c18da",
        "inputs_sha256": "d10f538dcc605a7a97347eb7e0ab4fe39997ca36b6ae09612ed2494fcbaf736f",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": None,
        "backward_through_sum": False,
        "run_under_no_grad": False,
    },
    {
        "name": "cpu_float32_heldout_guard_unary_metadata",
        "held_out": True,
        "category": "recompilation_guards",
        "program": "cpu_float32_heldout_guard_unary_metadata",
        "program_sha256": "dfb8f3cf4a5a8870598f8b7fd372f7532ef5b0c998c956e298137fc5cd39bd60",
        "make_inputs": "cpu_float32_heldout_guard_unary_inputs",
        "make_inputs_sha256": "412f3442c07dd3854c898a4bf877c499388627dcf8f99baca1c05ea4a3b3c3fb",
        "inputs_sha256": "114870c9149ff876c6980f96588c49925004138490843d0550cf96a833a27ba9",
        "arity": 1,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": 4,
    },
    {
        "name": "cpu_float32_heldout_guard_binary_metadata",
        "held_out": True,
        "category": "recompilation_guards",
        "program": "cpu_float32_heldout_guard_binary_metadata",
        "program_sha256": "6905320edf2d4f90693e960f646925ead3ab5753307263f22c4cd4d40ee5ccd2",
        "make_inputs": "cpu_float32_heldout_guard_binary_inputs",
        "make_inputs_sha256": "fe16ee7b769a065f56d79db8523c2b5dd922be322081d2cb4c2b9188f18d5b33",
        "inputs_sha256": "c0667a14246c1cc6213e22157429beb1263858e7babc82d8f687ee10d7423a30",
        "arity": 2,
        "fullgraph": True,
        "dynamic": None,
        "mode": None,
        "options": None,
        "recompile_limit": 4,
    },
)
EXPECTED_V11_GUARD_SCENARIO_MANIFEST = (
    {
        "name": "unary_shape_stride_requires_grad_guards",
        "held_out": False,
        "case_name": "cpu_float32_recompile_guard_unary_metadata",
        "steps": (
            {
                "name": "base",
                "make_inputs": "cpu_float32_recompile_guard_unary_inputs",
                "make_inputs_sha256": "ecac84421a6c706ffb6fd206231fa5ae0f706cb9f23145ca307a30b069a93070",
                "inputs_sha256": "35312be0abb470e0f8b34dadcbacd7effb9db9c88ee858c22b0174a1963c03e1",
                "guard_change": "initial",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "same_metadata",
                "make_inputs": "cpu_float32_recompile_guard_unary_same_metadata_inputs",
                "make_inputs_sha256": "c1c7d4647db3a6ede327e402497a8d64e14194351d3b7fb70775535e3e30bc70",
                "inputs_sha256": "54ae50380c5783c89d3b60b8e6f47c9061062cb3f9e37175e096d15ff33b17c4",
                "guard_change": "same_metadata",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "shape_change",
                "make_inputs": "cpu_float32_recompile_guard_unary_shape_inputs",
                "make_inputs_sha256": "952547561c750bd8210f71e0ef0c3e19af8d1912d8d1dfbbf3685302df7e5c18",
                "inputs_sha256": "01f1e1749b866372d2743834e99b57ed185c9ef4ed482f87bb23cb549d52ef99",
                "guard_change": "shape",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "stride_change",
                "make_inputs": "cpu_float32_recompile_guard_unary_stride_inputs",
                "make_inputs_sha256": "1e52c6eda609091cf2d8981ed92125a596a7cacd130403abe6741b560ad8f749",
                "inputs_sha256": "9857d132f7cd08b86cf70f048fa4b6ae34df30f8f5fee667b72c902acd1552fa",
                "guard_change": "stride",
                "expected_compile_count": 3,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "requires_grad_change",
                "make_inputs": "cpu_float32_recompile_guard_unary_requires_grad_inputs",
                "make_inputs_sha256": "f32d72665792485a72ce32cb68bf686e807b3a1136a974951454486f3f6064e6",
                "inputs_sha256": "50e73712a80b61031149846c8571af048d2f19c1fa77884d53e4fcc05bd79094",
                "guard_change": "requires_grad",
                "expected_compile_count": 4,
                "reset_before": False,
                "expect_limit_error": False,
            },
        ),
    },
    {
        "name": "binary_argument_metadata_guards",
        "held_out": False,
        "case_name": "cpu_float32_recompile_guard_binary_metadata",
        "steps": (
            {
                "name": "base",
                "make_inputs": "cpu_float32_recompile_guard_binary_inputs",
                "make_inputs_sha256": "78237478fbad18b34ba4e4fc1f4588d7476b9bed94060cc07a36b8cbb56fb22d",
                "inputs_sha256": "efde735c565edf1b5d5a841bca907124c22358cdcb27c7face8bed09dc4e78e7",
                "guard_change": "initial",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "same_metadata",
                "make_inputs": "cpu_float32_recompile_guard_binary_same_metadata_inputs",
                "make_inputs_sha256": "4302803f18bf8099584a4bd77c2a9bd4350e454275841ce58eb6e6811c5388a5",
                "inputs_sha256": "11cdc3528b6cf79529a547a9fd97348162c3441c0325e05c053ef668d70e20c8",
                "guard_change": "same_metadata",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "left_stride_change",
                "make_inputs": "cpu_float32_recompile_guard_binary_left_stride_inputs",
                "make_inputs_sha256": "f27fdecae1ad9f6da5254e5e721ec2f4cd7dfdaa1ba0775574903b62c04a47d2",
                "inputs_sha256": "49bacb0697f75abc1be61215a9f071b08a8479c1822377478ca9438fe3a41a8c",
                "guard_change": "stride",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "right_shape_change",
                "make_inputs": "cpu_float32_recompile_guard_binary_right_shape_inputs",
                "make_inputs_sha256": "557bb3af03674bc6985f790ec7b697c6d511925d1fbbdd9e5011221f469964fa",
                "inputs_sha256": "207e5c2cd0bc15c98eb1b38cfa5b82b8a6fc5a92291e2d9213a68c131a082615",
                "guard_change": "shape",
                "expected_compile_count": 3,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "right_requires_grad_change",
                "make_inputs": "cpu_float32_recompile_guard_binary_requires_grad_inputs",
                "make_inputs_sha256": "b3da29443e8223aa82ae1dbc64518c95af0cebcbe9c617c263ee67006215c66b",
                "inputs_sha256": "0fa4f8f29f47103549536bd6a6b2aa34c8bb864af6bc8fddd89df7e5eea74097",
                "guard_change": "requires_grad",
                "expected_compile_count": 4,
                "reset_before": False,
                "expect_limit_error": False,
            },
        ),
    },
    {
        "name": "requires_grad_branch_unary_cache",
        "held_out": False,
        "case_name": "cpu_float32_requires_grad_branch_unary",
        "steps": (
            {
                "name": "false_branch",
                "make_inputs": "cpu_float32_control_flow_requires_grad_false_inputs",
                "make_inputs_sha256": "fc6e0b56689ee7e83d8762fa86e52a3362e9456416818403c85799d45320f170",
                "inputs_sha256": "45bc0581ad4e4b64a4752682357d741919fd0a4a7bc62b1ddb4b74761fb98b64",
                "guard_change": "initial",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "same_false_metadata",
                "make_inputs": "cpu_float32_control_flow_requires_grad_false_inputs",
                "make_inputs_sha256": "fc6e0b56689ee7e83d8762fa86e52a3362e9456416818403c85799d45320f170",
                "inputs_sha256": "45bc0581ad4e4b64a4752682357d741919fd0a4a7bc62b1ddb4b74761fb98b64",
                "guard_change": "same_metadata",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "true_branch",
                "make_inputs": "cpu_float32_control_flow_requires_grad_true_inputs",
                "make_inputs_sha256": "5958cb3be400344b74dc3f8e50761aa3fc8207a8fd0ce698397df9d135ab0ae8",
                "inputs_sha256": "d28d2c7fba11ed6ce0f83af22ed44979aca58ccdc62900ba1099827577af7169",
                "guard_change": "requires_grad",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
        ),
    },
    {
        "name": "bounded_limit_then_reset",
        "held_out": False,
        "case_name": "cpu_float32_recompile_limit_reset",
        "steps": (
            {
                "name": "base",
                "make_inputs": "cpu_float32_recompile_guard_unary_inputs",
                "make_inputs_sha256": "ecac84421a6c706ffb6fd206231fa5ae0f706cb9f23145ca307a30b069a93070",
                "inputs_sha256": "35312be0abb470e0f8b34dadcbacd7effb9db9c88ee858c22b0174a1963c03e1",
                "guard_change": "initial",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "shape_change",
                "make_inputs": "cpu_float32_recompile_guard_unary_shape_inputs",
                "make_inputs_sha256": "952547561c750bd8210f71e0ef0c3e19af8d1912d8d1dfbbf3685302df7e5c18",
                "inputs_sha256": "01f1e1749b866372d2743834e99b57ed185c9ef4ed482f87bb23cb549d52ef99",
                "guard_change": "shape",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "limit_rejects_stride_change",
                "make_inputs": "cpu_float32_recompile_guard_unary_stride_inputs",
                "make_inputs_sha256": "1e52c6eda609091cf2d8981ed92125a596a7cacd130403abe6741b560ad8f749",
                "inputs_sha256": "9857d132f7cd08b86cf70f048fa4b6ae34df30f8f5fee667b72c902acd1552fa",
                "guard_change": "recompile_limit",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": True,
            },
            {
                "name": "cached_base_after_limit",
                "make_inputs": "cpu_float32_recompile_guard_unary_same_metadata_inputs",
                "make_inputs_sha256": "c1c7d4647db3a6ede327e402497a8d64e14194351d3b7fb70775535e3e30bc70",
                "inputs_sha256": "54ae50380c5783c89d3b60b8e6f47c9061062cb3f9e37175e096d15ff33b17c4",
                "guard_change": "same_metadata",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "reset_allows_stride_change",
                "make_inputs": "cpu_float32_recompile_guard_unary_stride_inputs",
                "make_inputs_sha256": "1e52c6eda609091cf2d8981ed92125a596a7cacd130403abe6741b560ad8f749",
                "inputs_sha256": "9857d132f7cd08b86cf70f048fa4b6ae34df30f8f5fee667b72c902acd1552fa",
                "guard_change": "reset",
                "expected_compile_count": 3,
                "reset_before": True,
                "expect_limit_error": False,
            },
        ),
    },
    {
        "name": "heldout_requires_grad_branch_binary_cache",
        "held_out": True,
        "case_name": "cpu_float32_heldout_requires_grad_branch_binary",
        "steps": (
            {
                "name": "false_branch",
                "make_inputs": "cpu_float32_heldout_control_flow_requires_grad_false_inputs",
                "make_inputs_sha256": "c4d0607c9d4758aea3e21110756b869a4f4276cb65c30c1cfd885b06410c18da",
                "inputs_sha256": "d10f538dcc605a7a97347eb7e0ab4fe39997ca36b6ae09612ed2494fcbaf736f",
                "guard_change": "initial",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "true_branch",
                "make_inputs": "cpu_float32_heldout_control_flow_requires_grad_true_inputs",
                "make_inputs_sha256": "be9f33ab1287cf4093e391c1bfe0e6ffd47305d98aa069f49629386bd3a295b0",
                "inputs_sha256": "12a9408be1920daf3d18e8002cea1cab121022b34f10b9222a67570ff608e6a0",
                "guard_change": "requires_grad",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "same_true_metadata",
                "make_inputs": "cpu_float32_heldout_control_flow_requires_grad_true_inputs",
                "make_inputs_sha256": "be9f33ab1287cf4093e391c1bfe0e6ffd47305d98aa069f49629386bd3a295b0",
                "inputs_sha256": "12a9408be1920daf3d18e8002cea1cab121022b34f10b9222a67570ff608e6a0",
                "guard_change": "same_metadata",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
        ),
    },
    {
        "name": "heldout_unary_rank3_metadata_mix",
        "held_out": True,
        "case_name": "cpu_float32_heldout_guard_unary_metadata",
        "steps": (
            {
                "name": "base",
                "make_inputs": "cpu_float32_heldout_guard_unary_inputs",
                "make_inputs_sha256": "412f3442c07dd3854c898a4bf877c499388627dcf8f99baca1c05ea4a3b3c3fb",
                "inputs_sha256": "114870c9149ff876c6980f96588c49925004138490843d0550cf96a833a27ba9",
                "guard_change": "initial",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "shape_change",
                "make_inputs": "cpu_float32_heldout_guard_unary_shape_inputs",
                "make_inputs_sha256": "590e01a3d6b9622c68966fca29be36ba92fcb48147de6bad3edab1d74e14e1af",
                "inputs_sha256": "67e43bbde994d906affcbf68eb31e7f237acc5c79d6190f89ad4814bcb786972",
                "guard_change": "shape",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "stride_and_requires_grad_change",
                "make_inputs": "cpu_float32_heldout_guard_unary_stride_requires_grad_inputs",
                "make_inputs_sha256": "577872feadf44c934a5624245e1fbc6595c0945172dbf8d8d4f8186f8297008c",
                "inputs_sha256": "59440bbd6be881be238076a51b40b627ffc9032e12e48392b0bf998c4bf5c76a",
                "guard_change": "stride_requires_grad",
                "expected_compile_count": 3,
                "reset_before": False,
                "expect_limit_error": False,
            },
        ),
    },
    {
        "name": "heldout_binary_broadcast_metadata_mix",
        "held_out": True,
        "case_name": "cpu_float32_heldout_guard_binary_metadata",
        "steps": (
            {
                "name": "base",
                "make_inputs": "cpu_float32_heldout_guard_binary_inputs",
                "make_inputs_sha256": "fe16ee7b769a065f56d79db8523c2b5dd922be322081d2cb4c2b9188f18d5b33",
                "inputs_sha256": "c0667a14246c1cc6213e22157429beb1263858e7babc82d8f687ee10d7423a30",
                "guard_change": "initial",
                "expected_compile_count": 1,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "shape_change",
                "make_inputs": "cpu_float32_heldout_guard_binary_shape_inputs",
                "make_inputs_sha256": "596962144a41758f9b85dbe98531a933a2f357ee5d34a61ab1f98f0aae021eae",
                "inputs_sha256": "fda184b91804be108254b6c4536c4025a2c145733ea4a57c31aefa6f22105b26",
                "guard_change": "shape",
                "expected_compile_count": 2,
                "reset_before": False,
                "expect_limit_error": False,
            },
            {
                "name": "requires_grad_change",
                "make_inputs": "cpu_float32_heldout_guard_binary_requires_grad_inputs",
                "make_inputs_sha256": "e98a36f47378c61ccae271408a54f6fc22f4d61f2d3a911364fd3a7bfb99e4c5",
                "inputs_sha256": "8d37546b609be936453b69281dfb1442aa96852213542dec97fb2b3b4b0b5ae0",
                "guard_change": "requires_grad",
                "expected_compile_count": 3,
                "reset_before": False,
                "expect_limit_error": False,
            },
        ),
    },
)


class EvaluationFatalError(RuntimeError):
    """Raised when the evaluator cannot produce a trustworthy score."""


@dataclass(frozen=True)
class CaseVerdict:
    name: str
    category: str
    held_out: bool
    passed: bool
    failure_kind: str | None = None
    message: str | None = None


def _load_compile_corpus_module():
    spec = importlib.util.spec_from_file_location(
        "_torch_rs_compile_corpus_for_evaluation",
        CORPUS_PATH,
    )
    if spec is None or spec.loader is None:
        raise EvaluationFatalError(f"could not load compile corpus from {CORPUS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise EvaluationFatalError(
            f"failed to import compile corpus {CORPUS_PATH}: {_exception_line(error)}"
        ) from error
    return module


def _version_without_local(version):
    return version.split("+", 1)[0]


def _exception_name(error):
    error_type = type(error)
    module = error_type.__module__
    name = error_type.__qualname__
    if module == "builtins":
        return name
    return f"{module}.{name}"


def _exception_line(error):
    return f"{_exception_name(error)}: {str(error).splitlines()[0]}"


def _package_version(distribution_name, module):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _all_cases(corpus_module, *, include_held_out):
    public_cases = tuple(getattr(corpus_module, "COMPILE_CORPUS", ()))
    if include_held_out:
        return public_cases + tuple(
            getattr(corpus_module, "COMPILE_HELD_OUT_CORPUS", ())
        )
    return public_cases


def _all_guard_scenarios(corpus_module, *, include_held_out):
    public_scenarios = tuple(
        getattr(corpus_module, "COMPILE_RECOMPILATION_GUARD_SCENARIOS", ())
    )
    if include_held_out:
        return public_scenarios + tuple(
            getattr(
                corpus_module,
                "COMPILE_HELD_OUT_RECOMPILATION_GUARD_SCENARIOS",
                (),
            )
        )
    return public_scenarios


def _case_by_name(corpus_module, case_name):
    for case in _all_cases(corpus_module, include_held_out=True):
        if getattr(case, "name", None) == case_name:
            return case
    raise EvaluationFatalError(f"guard scenario references unknown case {case_name!r}")


def _case_names(cases):
    return [case.name for case in cases]


def _guard_scenario_names(scenarios):
    return [scenario.name for scenario in scenarios]


def _json_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256(callable_object, *, errors, context):
    if not callable(callable_object):
        errors.append(f"{context} is not callable")
        return None
    try:
        source = inspect.getsource(callable_object)
    except (OSError, TypeError) as error:
        errors.append(f"{context} source is unavailable: {_exception_line(error)}")
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _callable_name(callable_object):
    return getattr(callable_object, "__name__", None)


def _instruction_global_name(instruction):
    name = instruction.argval
    if type(name) is str and name:
        return name
    argrepr = str(instruction.argrepr).strip()
    if argrepr.startswith("NULL + "):
        argrepr = argrepr[7:].strip()
    return argrepr or None


def _same_module_helper_functions(program):
    globals_dict = getattr(program, "__globals__", None)
    if type(globals_dict) is not dict:
        return ()
    module_name = getattr(program, "__module__", None)
    helpers = []
    seen = set()
    for instruction in dis.get_instructions(program):
        if instruction.opname != "LOAD_GLOBAL":
            continue
        name = _instruction_global_name(instruction)
        if name is None:
            continue
        helper = globals_dict.get(name)
        if type(helper) is not types.FunctionType:
            continue
        if helper.__globals__ is not globals_dict or helper.__module__ != module_name:
            continue
        if globals_dict.get(helper.__name__) is not helper:
            continue
        if helper in seen:
            continue
        seen.add(helper)
        helpers.append(helper)
    return tuple(helpers)


def _helper_manifest_entries(program, *, errors, context):
    return tuple(
        {
            "name": helper.__name__,
            "source_sha256": _source_sha256(
                helper,
                errors=errors,
                context=f"{context} helper {helper.__name__}",
            ),
        }
        for helper in _same_module_helper_functions(program)
    )


def _validate_module_level_callable(corpus_module, callable_object, *, context, errors):
    callable_name = _callable_name(callable_object)
    if type(callable_name) is not str or not callable_name:
        errors.append(f"{context} has invalid callable name {callable_name!r}")
        return callable_name
    if getattr(corpus_module, callable_name, None) is not callable_object:
        errors.append(
            f"{context} is not the module-level corpus function {callable_name!r}"
        )
    return callable_name


def _manifest_tensor_module(corpus_module, errors):
    tensor_module = getattr(corpus_module, "reference_torch", None)
    if tensor_module is None:
        tensor_module = getattr(corpus_module, "torch", None)
    if tensor_module is None:
        errors.append("compile corpus does not expose a tensor module")
    return tensor_module


def _inputs_sha256(make_inputs, tensor_module, *, errors, context):
    if tensor_module is None or not callable(make_inputs):
        return None
    try:
        return _json_sha256(_inputs_payload(make_inputs(tensor_module)))
    except Exception as error:
        errors.append(
            f"{context} input payload is not reproducible: {_exception_line(error)}"
        )
        return None


def _case_manifest_entry(corpus_module, case, *, held_out, tensor_module, errors):
    program = getattr(case, "program", None)
    make_inputs = getattr(case, "make_inputs", None)
    program_name = _validate_module_level_callable(
        corpus_module,
        program,
        context=f"{getattr(case, 'name', '<unnamed>')} program",
        errors=errors,
    )
    make_inputs_name = _validate_module_level_callable(
        corpus_module,
        make_inputs,
        context=f"{getattr(case, 'name', '<unnamed>')} input factory",
        errors=errors,
    )
    code = getattr(program, "__code__", None)
    return {
        "name": getattr(case, "name", None),
        "held_out": held_out,
        "category": getattr(case, "category", None),
        "program": program_name,
        "program_sha256": _source_sha256(
            program,
            errors=errors,
            context=f"{getattr(case, 'name', '<unnamed>')} program",
        ),
        "helper_sha256s": _helper_manifest_entries(
            program,
            errors=errors,
            context=f"{getattr(case, 'name', '<unnamed>')} program",
        ),
        "make_inputs": make_inputs_name,
        "make_inputs_sha256": _source_sha256(
            make_inputs,
            errors=errors,
            context=f"{getattr(case, 'name', '<unnamed>')} input factory",
        ),
        "inputs_sha256": _inputs_sha256(
            make_inputs,
            tensor_module,
            errors=errors,
            context=f"{getattr(case, 'name', '<unnamed>')} input factory",
        ),
        "arity": code.co_argcount if code is not None else None,
        "fullgraph": getattr(case, "fullgraph", None),
        "dynamic": getattr(case, "dynamic", None),
        "mode": getattr(case, "mode", None),
        "options": getattr(case, "options", None),
        "recompile_limit": getattr(case, "recompile_limit", None),
        "backward_through_sum": getattr(case, "backward_through_sum", None),
        "run_under_no_grad": getattr(case, "run_under_no_grad", None),
    }


def _guard_step_manifest_entry(corpus_module, scenario, step, *, tensor_module, errors):
    make_inputs = getattr(step, "make_inputs", None)
    make_inputs_name = _validate_module_level_callable(
        corpus_module,
        make_inputs,
        context=f"{getattr(scenario, 'name', '<unnamed>')}/{getattr(step, 'name', '<unnamed>')} input factory",
        errors=errors,
    )
    return {
        "name": getattr(step, "name", None),
        "make_inputs": make_inputs_name,
        "make_inputs_sha256": _source_sha256(
            make_inputs,
            errors=errors,
            context=f"{getattr(scenario, 'name', '<unnamed>')}/{getattr(step, 'name', '<unnamed>')} input factory",
        ),
        "inputs_sha256": _inputs_sha256(
            make_inputs,
            tensor_module,
            errors=errors,
            context=f"{getattr(scenario, 'name', '<unnamed>')}/{getattr(step, 'name', '<unnamed>')} input factory",
        ),
        "guard_change": getattr(step, "guard_change", None),
        "expected_compile_count": getattr(step, "expected_compile_count", None),
        "reset_before": getattr(step, "reset_before", None),
        "expect_limit_error": getattr(step, "expect_limit_error", None),
    }


def _guard_scenario_manifest_entry(
    corpus_module,
    scenario,
    *,
    held_out,
    tensor_module,
    errors,
):
    return {
        "name": getattr(scenario, "name", None),
        "held_out": held_out,
        "case_name": getattr(scenario, "case_name", None),
        "steps": tuple(
            _guard_step_manifest_entry(
                corpus_module,
                scenario,
                step,
                tensor_module=tensor_module,
                errors=errors,
            )
            for step in tuple(getattr(scenario, "steps", ()))
        ),
    }


def _compare_manifest_entry(actual, expected, *, context, errors):
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key == "steps" and isinstance(actual_value, tuple):
            expected_names = [step["name"] for step in expected_value]
            actual_names = [step.get("name") for step in actual_value]
            if actual_names != expected_names:
                errors.append(
                    f"{context} step names/order changed: "
                    f"{actual_names!r} != {expected_names!r}"
                )
            for actual_step, expected_step in zip(actual_value, expected_value):
                if actual_step.get("name") != expected_step["name"]:
                    continue
                _compare_manifest_entry(
                    actual_step,
                    expected_step,
                    context=f"{context}/{actual_step['name']}",
                    errors=errors,
                )
            continue
        if actual_value != expected_value:
            errors.append(
                f"{context} {key} changed: {actual_value!r} != {expected_value!r}"
            )


def _expected_case_manifest(held_out):
    return tuple(
        entry for entry in EXPECTED_V11_CASE_MANIFEST if entry["held_out"] is held_out
    )


def _expected_guard_scenario_manifest(held_out):
    return tuple(
        entry
        for entry in EXPECTED_V11_GUARD_SCENARIO_MANIFEST
        if entry["held_out"] is held_out
    )


def _validate_v11_case_manifest(
    corpus_module,
    cases,
    *,
    held_out,
    tensor_module,
    errors,
):
    label = "held-out" if held_out else "public"
    expected_entries = _expected_case_manifest(held_out)
    expected_names = [entry["name"] for entry in expected_entries]
    actual_names = [getattr(case, "name", None) for case in cases]
    if actual_names != expected_names:
        errors.append(
            f"{label} v11 case names/order changed: {actual_names!r} != {expected_names!r}"
        )

    expected_by_name = {entry["name"]: entry for entry in expected_entries}
    for case in cases:
        expected_entry = expected_by_name.get(getattr(case, "name", None))
        if expected_entry is None:
            continue
        actual_entry = _case_manifest_entry(
            corpus_module,
            case,
            held_out=held_out,
            tensor_module=tensor_module,
            errors=errors,
        )
        _compare_manifest_entry(
            actual_entry,
            expected_entry,
            context=f"{label} v11 case {case.name}",
            errors=errors,
        )


def _validate_v11_guard_scenario_manifest(
    corpus_module,
    scenarios,
    *,
    held_out,
    tensor_module,
    errors,
):
    label = "held-out" if held_out else "public"
    expected_entries = _expected_guard_scenario_manifest(held_out)
    expected_names = [entry["name"] for entry in expected_entries]
    actual_names = [getattr(scenario, "name", None) for scenario in scenarios]
    if actual_names != expected_names:
        errors.append(
            f"{label} v11 guard scenarios changed: {actual_names!r} != {expected_names!r}"
        )

    expected_by_name = {entry["name"]: entry for entry in expected_entries}
    for scenario in scenarios:
        expected_entry = expected_by_name.get(getattr(scenario, "name", None))
        if expected_entry is None:
            continue
        actual_entry = _guard_scenario_manifest_entry(
            corpus_module,
            scenario,
            held_out=held_out,
            tensor_module=tensor_module,
            errors=errors,
        )
        _compare_manifest_entry(
            actual_entry,
            expected_entry,
            context=f"{label} v11 guard scenario {scenario.name}",
            errors=errors,
        )


def _validate_corpus_metadata(corpus_module):
    errors = []
    version = getattr(corpus_module, "COMPILE_CORPUS_VERSION", None)
    if version != EXPECTED_CORPUS_VERSION:
        errors.append(
            f"corpus version mismatch: {version!r} != {EXPECTED_CORPUS_VERSION!r}"
        )

    category_weights = getattr(corpus_module, "CATEGORY_WEIGHTS", None)
    if not isinstance(category_weights, dict):
        errors.append("CATEGORY_WEIGHTS must be a dict")
        category_weights = {}
    else:
        if tuple(category_weights) != REQUIRED_CATEGORIES:
            errors.append(
                "CATEGORY_WEIGHTS categories changed or reordered: "
                f"{tuple(category_weights)!r}"
            )
        if category_weights != EXPECTED_CATEGORY_WEIGHTS:
            errors.append(
                "CATEGORY_WEIGHTS values changed: "
                f"{category_weights!r} != {EXPECTED_CATEGORY_WEIGHTS!r}"
            )
        if sum(category_weights.values()) != 100:
            errors.append(
                f"CATEGORY_WEIGHTS must sum to 100, got {sum(category_weights.values())}"
            )
        for category, weight in category_weights.items():
            if type(category) is not str or not category:
                errors.append(f"invalid category name {category!r}")
            if type(weight) is not int or weight <= 0:
                errors.append(f"{category!r} has invalid weight {weight!r}")

    public_cases = tuple(getattr(corpus_module, "COMPILE_CORPUS", ()))
    held_out_cases = tuple(getattr(corpus_module, "COMPILE_HELD_OUT_CORPUS", ()))
    if len(public_cases) != 21:
        errors.append(f"expected 21 public v11 cases, found {len(public_cases)}")
    if len(held_out_cases) != 13:
        errors.append(f"expected 13 held-out v11 cases, found {len(held_out_cases)}")

    seen_names = set()
    for case in (*public_cases, *held_out_cases):
        name = getattr(case, "name", None)
        category = getattr(case, "category", None)
        program = getattr(case, "program", None)
        make_inputs = getattr(case, "make_inputs", None)
        if type(name) is not str or not name:
            errors.append(f"case has invalid name {name!r}")
            continue
        if name in seen_names:
            errors.append(f"duplicate case name {name!r}")
        seen_names.add(name)
        if category not in category_weights:
            errors.append(f"{name} has unknown category {category!r}")
        if not callable(make_inputs):
            errors.append(f"{name} has non-callable input factory")
        code = getattr(program, "__code__", None)
        if code is None:
            errors.append(f"{name} program is not an exact Python function")
        elif code.co_argcount not in (1, 2):
            errors.append(f"{name} has unsupported arity {code.co_argcount}")
        if getattr(case, "fullgraph", None) is not True:
            errors.append(f"{name} must use fullgraph=True")
        for option_name in ("dynamic", "mode", "options"):
            if getattr(case, option_name, None) is not None:
                errors.append(f"{name} has unsupported {option_name!s} metadata")
        recompile_limit = getattr(case, "recompile_limit", None)
        if category == "recompilation_guards":
            if recompile_limit not in (2, 4):
                errors.append(f"{name} has invalid guard recompile_limit")
        elif recompile_limit is not None:
            errors.append(f"{name} has unexpected recompile_limit")
        backward_through_sum = getattr(case, "backward_through_sum", None)
        if type(backward_through_sum) is not bool:
            errors.append(f"{name} has invalid backward_through_sum metadata")
        if category == "training_autograd" and backward_through_sum is not True:
            errors.append(f"{name} must enable backward_through_sum")
        if category != "training_autograd" and backward_through_sum is not False:
            errors.append(f"{name} unexpectedly enables backward_through_sum")
        run_under_no_grad = getattr(case, "run_under_no_grad", None)
        if type(run_under_no_grad) is not bool:
            errors.append(f"{name} has invalid run_under_no_grad metadata")
        if category == "inference" and run_under_no_grad is not True:
            errors.append(f"{name} must run under torch.no_grad()")
        if category != "inference" and run_under_no_grad is not False:
            errors.append(f"{name} unexpectedly runs under torch.no_grad()")
        if run_under_no_grad and backward_through_sum:
            errors.append(f"{name} cannot both run no-grad and backward")

    public_scenarios = tuple(
        getattr(corpus_module, "COMPILE_RECOMPILATION_GUARD_SCENARIOS", ())
    )
    held_out_scenarios = tuple(
        getattr(corpus_module, "COMPILE_HELD_OUT_RECOMPILATION_GUARD_SCENARIOS", ())
    )
    if _guard_scenario_names(public_scenarios) != list(EXPECTED_PUBLIC_GUARD_SCENARIOS):
        errors.append("public recompilation guard scenarios do not match v11")
    if _guard_scenario_names(held_out_scenarios) != list(
        EXPECTED_HELD_OUT_GUARD_SCENARIOS
    ):
        errors.append("held-out recompilation guard scenarios do not match v11")
    for scenario in (*public_scenarios, *held_out_scenarios):
        scenario_name = getattr(scenario, "name", None)
        case_name = getattr(scenario, "case_name", None)
        steps = tuple(getattr(scenario, "steps", ()))
        if case_name not in seen_names:
            errors.append(f"{scenario_name} references unknown case {case_name!r}")
        if not steps:
            errors.append(f"{scenario_name} has no guard steps")
        last_compile_count = -1
        for step in steps:
            step_name = getattr(step, "name", None)
            guard_change = getattr(step, "guard_change", None)
            expected_count = getattr(step, "expected_compile_count", None)
            if type(step_name) is not str or not step_name:
                errors.append(f"{scenario_name} has invalid step name {step_name!r}")
            if type(guard_change) is not str or not guard_change:
                errors.append(f"{scenario_name}/{step_name} has invalid guard change")
            if type(expected_count) is not int or expected_count < last_compile_count:
                errors.append(
                    f"{scenario_name}/{step_name} has invalid expected compile count"
                )
            last_compile_count = expected_count if type(expected_count) is int else 0

    tensor_module = _manifest_tensor_module(corpus_module, errors)
    _validate_v11_case_manifest(
        corpus_module,
        public_cases,
        held_out=False,
        tensor_module=tensor_module,
        errors=errors,
    )
    _validate_v11_case_manifest(
        corpus_module,
        held_out_cases,
        held_out=True,
        tensor_module=tensor_module,
        errors=errors,
    )
    _validate_v11_guard_scenario_manifest(
        corpus_module,
        public_scenarios,
        held_out=False,
        tensor_module=tensor_module,
        errors=errors,
    )
    _validate_v11_guard_scenario_manifest(
        corpus_module,
        held_out_scenarios,
        held_out=True,
        tensor_module=tensor_module,
        errors=errors,
    )

    if errors:
        raise EvaluationFatalError(
            "malformed compile corpus metadata:\n" + "\n".join(errors)
        )


def _tensor_payload(tensor):
    return {
        "metadata": {
            "shape": list(tuple(tensor.shape)),
            "stride": list(tuple(tensor.stride())),
            "storage_offset": int(tensor.storage_offset()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "is_contiguous": bool(tensor.is_contiguous()),
        },
        "values": tensor.tolist(),
    }


def _is_output_container(value):
    return type(value) in (tuple, list)


def _output_payload(output):
    if _is_output_container(output):
        return {
            "container": type(output).__name__,
            "elements": [_output_payload(element) for element in output],
        }
    return _tensor_payload(output)


def _output_tensor_leaves(output):
    if _is_output_container(output):
        for element in output:
            yield from _output_tensor_leaves(element)
        return
    yield output


def _output_requires_grad(output):
    return any(tensor.requires_grad for tensor in _output_tensor_leaves(output))


def _output_sum(output):
    leaves = tuple(_output_tensor_leaves(output))
    if not leaves:
        raise AssertionError("compile corpus output must contain a Tensor leaf")
    total = leaves[0].sum()
    for leaf in leaves[1:]:
        total = total + leaf.sum()
    return total


def _inputs_payload(inputs):
    return [_tensor_payload(input) for input in inputs]


def _leaf_gradients_payload(inputs):
    gradients = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The \\.grad attribute of a Tensor that is not a leaf Tensor",
            category=UserWarning,
        )
        for input in inputs:
            gradient = getattr(input, "grad", None)
            if gradient is None:
                gradients.append(None)
            else:
                gradients.append(_tensor_payload(gradient))
    return gradients


def _case_run_under_no_grad(case):
    return getattr(case, "run_under_no_grad", False)


def _case_requires_alias_checks(case):
    return getattr(case, "category", None) == "mutation_aliasing_views"


def _storage_alias_payload(output, inputs):
    return [bool(output.is_set_to(input)) for input in inputs]


def _run_case_callable(module, case, callable_object, inputs):
    if _case_run_under_no_grad(case):
        with module.no_grad():
            return callable_object(*inputs)
    return callable_object(*inputs)


def _run_case_program(module, case, inputs):
    return _run_case_callable(module, case, case.program, inputs)


def _assert_payload_match(actual, expected, *, label):
    if actual != expected:
        raise AssertionError(
            f"{label} observable mismatch: actual={actual!r}, expected={expected!r}"
        )


def _compile_kwargs_from_case(case, backend):
    kwargs = {
        "backend": backend,
        "fullgraph": case.fullgraph,
    }
    if case.dynamic is not None:
        kwargs["dynamic"] = case.dynamic
    if case.mode is not None:
        kwargs["mode"] = case.mode
    if case.options is not None:
        kwargs["options"] = dict(case.options)
    if case.recompile_limit is not None:
        kwargs["recompile_limit"] = case.recompile_limit
    return kwargs


def _make_recording_backend(calls):
    def backend(graph_module, example_inputs):
        calls.append((graph_module, example_inputs))
        return graph_module.forward

    return backend


def _reset_reference_compile_state(reference_torch):
    dynamo = getattr(reference_torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if reset is not None:
        reset()


def _reference_case_result(reference_torch, case):
    _reset_reference_compile_state(reference_torch)
    backend_calls = []
    backend = _make_recording_backend(backend_calls)
    backward_through_sum = getattr(case, "backward_through_sum", False)
    run_under_no_grad = _case_run_under_no_grad(case)
    check_aliasing = _case_requires_alias_checks(case)
    check_input_gradients = run_under_no_grad or check_aliasing
    inputs = case.make_inputs(reference_torch)
    before_inputs = _inputs_payload(inputs)
    before_gradients = _leaf_gradients_payload(inputs)
    expected_inputs = case.make_inputs(reference_torch)
    expected_before_gradients = _leaf_gradients_payload(expected_inputs)
    expected = _run_case_program(reference_torch, case, expected_inputs)
    expected_gradients = None
    if backward_through_sum:
        _output_sum(expected).backward()
        expected_gradients = _leaf_gradients_payload(expected_inputs)
    if check_input_gradients:
        _assert_payload_match(
            _leaf_gradients_payload(expected_inputs),
            expected_before_gradients,
            label=f"{case.name}/reference_expected_input_gradients",
        )
    compiled = reference_torch.compile(
        case.program,
        **_compile_kwargs_from_case(case, backend),
    )
    actual = _run_case_callable(reference_torch, case, compiled, inputs)
    after_forward_inputs = _inputs_payload(inputs)

    _assert_payload_match(
        _output_payload(actual),
        _output_payload(expected),
        label=f"{case.name}/reference",
    )
    _assert_payload_match(
        after_forward_inputs,
        before_inputs,
        label=f"{case.name}/inputs",
    )
    input_gradients = None
    if check_input_gradients:
        input_gradients = _leaf_gradients_payload(inputs)
        _assert_payload_match(
            input_gradients,
            before_gradients,
            label=f"{case.name}/reference_input_gradients",
        )
    if check_aliasing:
        _assert_payload_match(
            _storage_alias_payload(actual, inputs),
            _storage_alias_payload(expected, expected_inputs),
            label=f"{case.name}/reference_storage_aliasing",
        )
        if actual.requires_grad:
            raise AssertionError(f"{case.name} detach reference output requires grad")
    if run_under_no_grad:
        if _output_requires_grad(actual):
            raise AssertionError(f"{case.name} no-grad reference output requires grad")
    actual_gradients = None
    if backward_through_sum:
        _output_sum(actual).backward()
        actual_gradients = _leaf_gradients_payload(inputs)
        _assert_payload_match(
            actual_gradients,
            expected_gradients,
            label=f"{case.name}/reference_backward_sum",
        )
        _assert_payload_match(
            _inputs_payload(inputs),
            before_inputs,
            label=f"{case.name}/inputs_after_backward",
        )
    if len(backend_calls) < 1:
        raise AssertionError(f"{case.name} did not invoke the reference backend")
    result = {
        "name": case.name,
        "category": case.category,
        "output": _output_payload(actual),
        "input_count": len(inputs),
        "backend_call_count": len(backend_calls),
    }
    if backward_through_sum:
        result["leaf_gradients"] = actual_gradients
    if check_input_gradients:
        result["input_gradients"] = input_gradients
    if check_aliasing:
        result["output_aliases_inputs"] = _storage_alias_payload(actual, inputs)
    return result


def _reference_guard_scenario_result(corpus_module, reference_torch, scenario):
    _reset_reference_compile_state(reference_torch)
    backend_calls = []
    backend = _make_recording_backend(backend_calls)
    case = _case_by_name(corpus_module, scenario.case_name)
    compiled = reference_torch.compile(
        case.program,
        **_compile_kwargs_from_case(case, backend),
    )
    steps = []

    for step in scenario.steps:
        if step.reset_before:
            _reset_reference_compile_state(reference_torch)
        inputs = step.make_inputs(reference_torch)
        before_inputs = _inputs_payload(inputs)
        if step.expect_limit_error:
            try:
                _run_case_callable(reference_torch, case, compiled, inputs)
            except Exception as error:
                steps.append(
                    {
                        "name": step.name,
                        "status": "expected_error",
                        "guard_change": step.guard_change,
                        "error_type": _exception_name(error),
                        "error_message": str(error).splitlines()[0],
                    }
                )
                continue
            raise AssertionError(
                f"{scenario.name}/{step.name} did not raise the expected limit error"
            )

        expected = _run_case_program(
            reference_torch,
            case,
            step.make_inputs(reference_torch),
        )
        actual = _run_case_callable(reference_torch, case, compiled, inputs)
        after_inputs = _inputs_payload(inputs)
        _assert_payload_match(
            _output_payload(actual),
            _output_payload(expected),
            label=f"{scenario.name}/{step.name}/reference",
        )
        _assert_payload_match(
            after_inputs,
            before_inputs,
            label=f"{scenario.name}/{step.name}/inputs",
        )
        steps.append(
            {
                "name": step.name,
                "status": "ok",
                "guard_change": step.guard_change,
                "output": _output_payload(actual),
            }
        )

    if len(backend_calls) < 1:
        raise AssertionError(f"{scenario.name} did not invoke the reference backend")
    return {
        "name": scenario.name,
        "case": case.name,
        "backend_call_count": len(backend_calls),
        "steps": steps,
    }


def _reference_results(corpus_module, reference_torch, cases, scenarios):
    case_results = {}
    scenario_results = {}
    for case in cases:
        try:
            case_results[case.name] = _reference_case_result(reference_torch, case)
        except Exception as error:
            raise EvaluationFatalError(
                f"reference PyTorch rejected the eligible case {case.name}: "
                f"{_exception_line(error)}"
            ) from error
    for scenario in scenarios:
        try:
            scenario_results[scenario.name] = _reference_guard_scenario_result(
                corpus_module,
                reference_torch,
                scenario,
            )
        except Exception as error:
            raise EvaluationFatalError(
                "reference PyTorch rejected the eligible guard scenario "
                f"{scenario.name}: {_exception_line(error)}"
            ) from error
    return case_results, scenario_results


class _PytorchImportBlocker:
    def __init__(self, exception_type):
        self.exception_type = exception_type

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "torch" or fullname.startswith("torch."):
            raise self.exception_type(
                "torch import is blocked during torch_rs candidate execution"
            )
        return None


def _drop_pytorch_modules():
    return {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "torch" or name.startswith("torch.")
    }


@contextmanager
def _blocked_pytorch_imports(
    exception_type=ImportError,
    *,
    context="blocked PyTorch import context",
):
    removed = _drop_pytorch_modules()
    blocker = _PytorchImportBlocker(exception_type)
    sys.meta_path.insert(0, blocker)
    try:
        yield
    except Exception:
        raise
    else:
        _assert_no_pytorch_modules(context)
    finally:
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name)
        sys.modules.update(removed)


def _assert_no_pytorch_modules(context):
    imported = sorted(
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    )
    if imported:
        raise AssertionError(
            f"{context} imported installed PyTorch modules: {', '.join(imported[:8])}"
        )


@contextmanager
def _candidate_compile_counters():
    from torch_rs import _compile_bytecode
    from torch_rs import _compile_trace

    original_lower_compile_graph = _compile_bytecode.lower_compile_graph
    original_execute_compile_trace_graph = _compile_trace.execute_compile_trace_graph
    counters = {
        "lower_compile_graph": 0,
        "execute_compile_trace_graph": 0,
    }

    def counting_lower_compile_graph(
        program,
        input_metadatas,
        *,
        name=None,
        compile_request=None,
    ):
        counters["lower_compile_graph"] += 1
        return original_lower_compile_graph(
            program,
            input_metadatas,
            name=name,
            compile_request=compile_request,
        )

    def counting_execute_compile_trace_graph(graph, *inputs):
        counters["execute_compile_trace_graph"] += 1
        return original_execute_compile_trace_graph(graph, *inputs)

    _compile_bytecode.lower_compile_graph = counting_lower_compile_graph
    _compile_trace.execute_compile_trace_graph = counting_execute_compile_trace_graph
    try:
        yield counters
    finally:
        _compile_bytecode.lower_compile_graph = original_lower_compile_graph
        _compile_trace.execute_compile_trace_graph = original_execute_compile_trace_graph


@contextmanager
def _program_call_counter(*programs):
    old_profile = sys.getprofile()
    codes = {
        program.__code__
        for program in programs
        if getattr(program, "__code__", None) is not None
    }
    calls = {"count": 0}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            calls["count"] += 1
        if old_profile is not None:
            old_profile(frame, event, arg)
        return profile

    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(old_profile)


def _candidate_case_result(corpus_module, case):
    torch_rs = corpus_module.torch
    torch_rs.compiler.reset()
    backward_through_sum = getattr(case, "backward_through_sum", False)
    run_under_no_grad = _case_run_under_no_grad(case)
    check_aliasing = _case_requires_alias_checks(case)
    check_input_gradients = run_under_no_grad or check_aliasing
    expected_inputs = case.make_inputs(torch_rs)
    expected_before_gradients = _leaf_gradients_payload(expected_inputs)
    expected = _run_case_program(torch_rs, case, expected_inputs)
    expected_gradients = None
    if backward_through_sum:
        _output_sum(expected).backward()
        expected_gradients = _leaf_gradients_payload(expected_inputs)
    if check_input_gradients:
        _assert_payload_match(
            _leaf_gradients_payload(expected_inputs),
            expected_before_gradients,
            label=f"{case.name}/torch_rs_expected_input_gradients",
        )
    inputs = case.make_inputs(torch_rs)
    before_inputs = _inputs_payload(inputs)
    before_gradients = _leaf_gradients_payload(inputs)
    user_callables = (case.program, *_same_module_helper_functions(case.program))
    with _candidate_compile_counters() as counters:
        compiled = torch_rs.compile(
            case.program,
            **_compile_kwargs_from_case(case, "eager"),
        )
        if getattr(compiled, "_torch_rs_compile_backend", None) != "eager":
            raise AssertionError(f"{case.name} did not resolve backend='eager'")
        with _program_call_counter(*user_callables) as program_calls:
            actual = _run_case_callable(torch_rs, case, compiled, inputs)
        if program_calls["count"] != 0:
            raise AssertionError(
                f"{case.name} executed original Python user code during compiled call"
            )
        if counters["lower_compile_graph"] != 1:
            raise AssertionError(
                f"{case.name} compiled call lowered {counters['lower_compile_graph']} "
                "graphs, expected exactly 1"
            )
        if counters["execute_compile_trace_graph"] != 1:
            raise AssertionError(
                f"{case.name} executed {counters['execute_compile_trace_graph']} "
                "native trace graphs, expected exactly 1"
            )
        after_forward_inputs = _inputs_payload(inputs)

        second_inputs = case.make_inputs(torch_rs)
        before_second_inputs = _inputs_payload(second_inputs)
        before_second_gradients = _leaf_gradients_payload(second_inputs)
        with _program_call_counter(*user_callables) as second_program_calls:
            second_actual = _run_case_callable(
                torch_rs,
                case,
                compiled,
                second_inputs,
            )
        if second_program_calls["count"] != 0:
            raise AssertionError(
                f"{case.name} executed original Python user code on cache hit"
            )
        if counters["lower_compile_graph"] != 1:
            raise AssertionError(f"{case.name} did not reuse the compiled cache")
        if counters["execute_compile_trace_graph"] != 2:
            raise AssertionError(
                f"{case.name} cache hit did not execute the native trace graph"
            )

    output = _output_payload(actual)
    _assert_payload_match(
        output,
        _output_payload(expected),
        label=f"{case.name}/torch_rs",
    )
    _assert_payload_match(
        _output_payload(second_actual),
        _output_payload(expected),
        label=f"{case.name}/torch_rs/cache_hit",
    )
    _assert_payload_match(
        after_forward_inputs,
        before_inputs,
        label=f"{case.name}/inputs",
    )
    _assert_payload_match(
        _inputs_payload(second_inputs),
        before_second_inputs,
        label=f"{case.name}/cache_hit_inputs",
    )
    input_gradients = None
    if run_under_no_grad and (
        _output_requires_grad(actual) or _output_requires_grad(second_actual)
    ):
        raise AssertionError(f"{case.name} no-grad candidate output requires grad")
    if check_input_gradients:
        input_gradients = _leaf_gradients_payload(inputs)
        _assert_payload_match(
            input_gradients,
            before_gradients,
            label=f"{case.name}/input_gradients",
        )
        _assert_payload_match(
            _leaf_gradients_payload(second_inputs),
            before_second_gradients,
            label=f"{case.name}/cache_hit_input_gradients",
        )
    if check_aliasing:
        _assert_payload_match(
            _storage_alias_payload(actual, inputs),
            _storage_alias_payload(expected, expected_inputs),
            label=f"{case.name}/storage_aliasing",
        )
        _assert_payload_match(
            _storage_alias_payload(second_actual, second_inputs),
            _storage_alias_payload(expected, expected_inputs),
            label=f"{case.name}/cache_hit_storage_aliasing",
        )
    actual_gradients = None
    if backward_through_sum:
        _output_sum(actual).backward()
        actual_gradients = _leaf_gradients_payload(inputs)
        _assert_payload_match(
            actual_gradients,
            expected_gradients,
            label=f"{case.name}/torch_rs_backward_sum",
        )
        _assert_payload_match(
            _inputs_payload(inputs),
            before_inputs,
            label=f"{case.name}/inputs_after_backward",
        )

        _output_sum(second_actual).backward()
        _assert_payload_match(
            _leaf_gradients_payload(second_inputs),
            expected_gradients,
            label=f"{case.name}/torch_rs_cache_hit_backward_sum",
        )
        _assert_payload_match(
            _inputs_payload(second_inputs),
            before_second_inputs,
            label=f"{case.name}/cache_hit_inputs_after_backward",
        )
    _assert_no_pytorch_modules(case.name)
    result = {
        "name": case.name,
        "category": case.category,
        "status": "passed",
        "output": output,
        "lower_compile_graph_count": counters["lower_compile_graph"],
        "execute_compile_trace_graph_count": counters["execute_compile_trace_graph"],
    }
    if backward_through_sum:
        result["leaf_gradients"] = actual_gradients
    if check_input_gradients:
        result["input_gradients"] = input_gradients
    if check_aliasing:
        result["output_aliases_inputs"] = _storage_alias_payload(actual, inputs)
    return result


def _candidate_guard_scenario_result(corpus_module, scenario):
    torch_rs = corpus_module.torch
    case = _case_by_name(corpus_module, scenario.case_name)
    torch_rs.compiler.reset()
    steps = []
    with _candidate_compile_counters() as counters:
        compiled = torch_rs.compile(
            case.program,
            **_compile_kwargs_from_case(case, "eager"),
        )
        if getattr(compiled, "_torch_rs_compile_backend", None) != "eager":
            raise AssertionError(f"{scenario.name} did not resolve backend='eager'")
        user_callables = (case.program, *_same_module_helper_functions(case.program))
        for step in scenario.steps:
            if step.reset_before:
                torch_rs.compiler.reset()
            inputs = step.make_inputs(torch_rs)
            before_inputs = _inputs_payload(inputs)
            previous_execute_count = counters["execute_compile_trace_graph"]

            if step.expect_limit_error:
                with _program_call_counter(*user_callables) as program_calls:
                    try:
                        _run_case_callable(torch_rs, case, compiled, inputs)
                    except Exception as error:
                        if f"recompile_limit={case.recompile_limit}" not in str(error):
                            raise AssertionError(
                                f"{scenario.name}/{step.name} raised the wrong "
                                f"limit error: {_exception_line(error)}"
                            ) from error
                    else:
                        raise AssertionError(
                            f"{scenario.name}/{step.name} did not raise the "
                            "expected limit error"
                        )
                if program_calls["count"] != 0:
                    raise AssertionError(
                        f"{scenario.name}/{step.name} executed the original "
                        "Python user code while rejecting a metadata miss"
                    )
                if counters["lower_compile_graph"] != step.expected_compile_count:
                    raise AssertionError(
                        f"{scenario.name}/{step.name} lower count "
                        f"{counters['lower_compile_graph']} != "
                        f"{step.expected_compile_count}"
                    )
                if counters["execute_compile_trace_graph"] != previous_execute_count:
                    raise AssertionError(
                        f"{scenario.name}/{step.name} executed the native trace graph "
                        "while rejecting a metadata miss"
                    )
                steps.append(
                    {
                        "name": step.name,
                        "status": "expected_error",
                        "guard_change": step.guard_change,
                        "lower_compile_graph_count": counters[
                            "lower_compile_graph"
                        ],
                    }
                )
                continue

            expected = _run_case_program(torch_rs, case, step.make_inputs(torch_rs))
            with _program_call_counter(*user_callables) as program_calls:
                actual = _run_case_callable(torch_rs, case, compiled, inputs)
            if program_calls["count"] != 0:
                raise AssertionError(
                    f"{scenario.name}/{step.name} executed the original Python "
                    "user code during compiled call"
                )
            after_inputs = _inputs_payload(inputs)
            _assert_payload_match(
                _output_payload(actual),
                _output_payload(expected),
                label=f"{scenario.name}/{step.name}/torch_rs",
            )
            _assert_payload_match(
                after_inputs,
                before_inputs,
                label=f"{scenario.name}/{step.name}/inputs",
            )
            if counters["lower_compile_graph"] != step.expected_compile_count:
                raise AssertionError(
                    f"{scenario.name}/{step.name} lower count "
                    f"{counters['lower_compile_graph']} != {step.expected_compile_count}"
                )
            expected_execute_count = previous_execute_count + 1
            if counters["execute_compile_trace_graph"] != expected_execute_count:
                raise AssertionError(
                    f"{scenario.name}/{step.name} execute count "
                    f"{counters['execute_compile_trace_graph']} != "
                    f"{expected_execute_count}"
                )
            steps.append(
                {
                    "name": step.name,
                    "status": "ok",
                    "guard_change": step.guard_change,
                    "output": _output_payload(actual),
                    "lower_compile_graph_count": counters["lower_compile_graph"],
                    "execute_compile_trace_graph_count": counters[
                        "execute_compile_trace_graph"
                    ],
                }
            )
    _assert_no_pytorch_modules(scenario.name)
    return {
        "name": scenario.name,
        "case": case.name,
        "status": "passed",
        "steps": steps,
    }


def _failed_candidate_result(name, category, error):
    return {
        "name": name,
        "category": category,
        "status": "failed",
        "failure_kind": _exception_name(error),
        "message": str(error).splitlines()[0],
    }


def _run_candidate_worker(args):
    try:
        with _blocked_pytorch_imports(
            ImportError,
            context="candidate metadata validation",
        ):
            corpus_module = _load_compile_corpus_module()
            _validate_corpus_metadata(corpus_module)
        cases = _select_subset(corpus_module, args.subset)
        scenarios = _select_guard_subset(corpus_module, args.subset)
        case_results = []
        scenario_results = []
        with _blocked_pytorch_imports(
            RuntimeError,
            context="candidate execution",
        ):
            _assert_no_pytorch_modules("candidate worker startup")
            for case in cases:
                try:
                    case_results.append(_candidate_case_result(corpus_module, case))
                except Exception as error:
                    case_results.append(
                        _failed_candidate_result(case.name, case.category, error)
                    )
            for scenario in scenarios:
                try:
                    scenario_results.append(
                        _candidate_guard_scenario_result(corpus_module, scenario)
                    )
                except Exception as error:
                    scenario_results.append(
                        {
                            "name": scenario.name,
                            "case": scenario.case_name,
                            "status": "failed",
                            "failure_kind": _exception_name(error),
                            "message": str(error).splitlines()[0],
                        }
                    )
        torch_rs = corpus_module.torch
        return {
            "ok": True,
            "environment": {
                "torch_rs_version": _package_version("torch-rs", torch_rs),
                "torch_rs_path": getattr(torch_rs, "__file__", None),
                "python": sys.version.replace("\n", " "),
                "python_executable": sys.executable,
            },
            "cases": case_results,
            "guard_scenarios": scenario_results,
        }
    except Exception as error:
        return {
            "ok": False,
            "fatal_error": _exception_line(error),
            "traceback": traceback.format_exc(limit=8),
        }


def _run_candidate_subprocess(subset):
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--candidate-worker",
            "--subset",
            subset,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationFatalError(
            "torch_rs candidate worker did not emit valid JSON: "
            f"exit={completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}"
        ) from error
    if completed.returncode != 0:
        raise EvaluationFatalError(
            "torch_rs candidate worker failed before scoring: "
            f"exit={completed.returncode}, payload={payload!r}, "
            f"stderr={completed.stderr!r}"
        )
    return payload


def _select_subset(corpus_module, subset):
    if subset == "full":
        return _all_cases(corpus_module, include_held_out=True)
    if subset == "public":
        return _all_cases(corpus_module, include_held_out=False)
    raise ValueError(f"unknown subset {subset!r}")


def _select_guard_subset(corpus_module, subset):
    if subset == "full":
        return _all_guard_scenarios(corpus_module, include_held_out=True)
    if subset == "public":
        return _all_guard_scenarios(corpus_module, include_held_out=False)
    raise ValueError(f"unknown subset {subset!r}")


def _case_is_held_out(corpus_module, case_name):
    return case_name in {case.name for case in corpus_module.COMPILE_HELD_OUT_CORPUS}


def _score_case_verdicts(category_weights, verdicts):
    by_category = {
        category: {"eligible": 0, "passed": 0, "failed": 0}
        for category in category_weights
    }
    for verdict in verdicts:
        row = by_category[verdict.category]
        row["eligible"] += 1
        if verdict.passed:
            row["passed"] += 1
        else:
            row["failed"] += 1

    category_scores = {}
    total_score = 0.0
    for category, weight in category_weights.items():
        row = by_category[category]
        if row["eligible"] == 0:
            ratio = 0.0
        else:
            ratio = row["passed"] / row["eligible"]
        weighted_score = weight * ratio
        total_score += weighted_score
        category_scores[category] = {
            "weight": weight,
            "eligible": row["eligible"],
            "passed": row["passed"],
            "failed": row["failed"],
            "weighted_score": weighted_score,
        }
    return total_score, category_scores


def _compare_worker_to_reference(
    corpus_module,
    cases,
    reference_case_results,
    reference_scenario_results,
    worker_payload,
):
    if not worker_payload.get("ok"):
        return [
            CaseVerdict(
                case.name,
                case.category,
                _case_is_held_out(corpus_module, case.name),
                False,
                "candidate_import_or_environment_failure",
                worker_payload.get("fatal_error", "candidate worker failed"),
            )
            for case in cases
        ], []

    worker_cases = {row.get("name"): row for row in worker_payload.get("cases", [])}
    worker_scenarios = {
        row.get("name"): row for row in worker_payload.get("guard_scenarios", [])
    }
    guard_failure_by_case = defaultdict(list)
    for scenario_name, reference_scenario in reference_scenario_results.items():
        candidate_scenario = worker_scenarios.get(scenario_name)
        if candidate_scenario is None:
            guard_failure_by_case[reference_scenario["case"]].append(
                f"{scenario_name} missing from candidate worker output"
            )
            continue
        if candidate_scenario.get("status") != "passed":
            guard_failure_by_case[reference_scenario["case"]].append(
                f"{scenario_name} failed: {candidate_scenario.get('message')}"
            )
            continue
        candidate_steps = {
            step.get("name"): step for step in candidate_scenario["steps"]
        }
        for reference_step in reference_scenario["steps"]:
            candidate_step = candidate_steps.get(reference_step["name"])
            if candidate_step is None:
                guard_failure_by_case[reference_scenario["case"]].append(
                    f"{scenario_name}/{reference_step['name']} missing"
                )
                continue
            if candidate_step.get("status") != reference_step["status"]:
                guard_failure_by_case[reference_scenario["case"]].append(
                    f"{scenario_name}/{reference_step['name']} status "
                    f"{candidate_step.get('status')!r} != {reference_step['status']!r}"
                )
                continue
            if reference_step["status"] == "ok":
                try:
                    _assert_payload_match(
                        candidate_step.get("output"),
                        reference_step.get("output"),
                        label=f"{scenario_name}/{reference_step['name']}",
                    )
                except AssertionError as error:
                    guard_failure_by_case[reference_scenario["case"]].append(str(error))

    verdicts = []
    for case in cases:
        reference_case = reference_case_results[case.name]
        candidate_case = worker_cases.get(case.name)
        held_out = _case_is_held_out(corpus_module, case.name)
        if candidate_case is None:
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    "candidate_missing_case",
                    "candidate worker did not report this case",
                )
            )
            continue
        if candidate_case.get("status") != "passed":
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    candidate_case.get("failure_kind", "candidate_failure"),
                    candidate_case.get("message", "candidate failed"),
                )
            )
            continue
        try:
            _assert_payload_match(
                candidate_case.get("output"),
                reference_case.get("output"),
                label=case.name,
            )
        except AssertionError as error:
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    "reference_mismatch",
                    str(error),
                )
            )
            continue
        if getattr(case, "backward_through_sum", False):
            try:
                _assert_payload_match(
                    candidate_case.get("leaf_gradients"),
                    reference_case.get("leaf_gradients"),
                    label=f"{case.name}/leaf_gradients",
                )
            except AssertionError as error:
                verdicts.append(
                    CaseVerdict(
                        case.name,
                        case.category,
                        held_out,
                        False,
                        "reference_mismatch",
                        str(error),
                    )
                )
                continue
        if _case_run_under_no_grad(case) or _case_requires_alias_checks(case):
            try:
                _assert_payload_match(
                    candidate_case.get("input_gradients"),
                    reference_case.get("input_gradients"),
                    label=f"{case.name}/input_gradients",
                )
            except AssertionError as error:
                verdicts.append(
                    CaseVerdict(
                        case.name,
                        case.category,
                        held_out,
                        False,
                        "reference_mismatch",
                        str(error),
                    )
                )
                continue
        if _case_requires_alias_checks(case):
            try:
                _assert_payload_match(
                    candidate_case.get("output_aliases_inputs"),
                    reference_case.get("output_aliases_inputs"),
                    label=f"{case.name}/output_aliases_inputs",
                )
            except AssertionError as error:
                verdicts.append(
                    CaseVerdict(
                        case.name,
                        case.category,
                        held_out,
                        False,
                        "reference_mismatch",
                        str(error),
                    )
                )
                continue
        guard_failures = guard_failure_by_case.get(case.name)
        if guard_failures:
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    "guard_scenario_failure",
                    "; ".join(guard_failures),
                )
            )
            continue
        verdicts.append(CaseVerdict(case.name, case.category, held_out, True))
    return verdicts, worker_payload.get("guard_scenarios", [])


def _category_line(category, row):
    return (
        f"{category} weight {row['weight']}: "
        f"{row['passed']}/{row['eligible']} eligible passed"
        if row["eligible"]
        else f"{category} weight {row['weight']}: 0 eligible, zero credit"
    )


def _summarize_output(
    corpus_module,
    reference_torch,
    worker_payload,
    verdicts,
    category_scores,
    score,
    subset,
):
    passed = sum(1 for verdict in verdicts if verdict.passed)
    failed = len(verdicts) - passed
    categories_with_cases = [
        category for category, row in category_scores.items() if row["eligible"]
    ]
    summary = (
        f"torch.compile coverage parity is {score:.1f}/100 from "
        f"{passed}/{len(verdicts)} reference-eligible {subset} corpus cases. "
        f"Covered categories: {', '.join(categories_with_cases)}; all other "
        "weighted categories remain zero-credit until they have passing "
        "reference-compilable torch_rs cases."
    )
    if failed:
        summary = (
            f"torch.compile coverage parity is {score:.1f}/100 with "
            f"{failed} failed reference-eligible {subset} corpus cases. "
            "Failed cases receive zero credit."
        )

    public_names = _case_names(corpus_module.COMPILE_CORPUS)
    held_out_names = _case_names(corpus_module.COMPILE_HELD_OUT_CORPUS)
    environment = worker_payload.get("environment", {}) if worker_payload else {}
    evidence = [
        (
            f"{EVALUATOR_VERSION} executed corpus {EXPECTED_CORPUS_VERSION} "
            f"with category weights summing to {sum(corpus_module.CATEGORY_WEIGHTS.values())}."
        ),
        (
            f"Reference PyTorch {_version_without_local(reference_torch.__version__)} "
            f"from {getattr(reference_torch, '__file__', None)} compiled and ran "
            f"all {len(verdicts)} selected eligible cases before scoring."
        ),
        (
            f"torch_rs from {environment.get('torch_rs_path')} ran in a subprocess "
            "with installed PyTorch imports blocked and native lowerer/executor "
            "calls instrumented."
        ),
        (
            f"Public cases: {', '.join(public_names)}. Held-out cases: "
            f"{', '.join(held_out_names)}."
        ),
        "; ".join(_category_line(category, row) for category, row in category_scores.items()),
        (
            "Recompilation guards validated: "
            f"{', '.join(_guard_scenario_names(_select_guard_subset(corpus_module, subset)))}."
        ),
    ]
    failures = [verdict for verdict in verdicts if not verdict.passed]
    for verdict in failures[:5]:
        evidence.append(
            f"{verdict.name} failed ({verdict.failure_kind}): {verdict.message}"
        )
    if len(failures) > 5:
        evidence.append(f"{len(failures) - 5} additional case failures omitted.")

    zero_credit_categories = [
        category for category, row in category_scores.items() if row["eligible"] == 0
    ]
    suggestions = []
    if zero_credit_categories:
        suggestions.append(
            "Add native compile support and reference-eligible corpus cases for "
            + ", ".join(zero_credit_categories[:6])
            + ("." if len(zero_credit_categories) <= 6 else ", and remaining categories.")
        )
    if failures:
        suggestions.append(
            "Fix the failing torch_rs compile cases before claiming coverage for "
            "their weighted categories."
        )
    suggestions.append(
        "Keep any quick screen as a strict subset of this corpus; run the default "
        "full evaluator for final scoring."
    )
    return {
        "score": round(score, 6),
        "summary": summary,
        "evidence": evidence,
        "suggestions": suggestions,
    }


def evaluate(subset):
    corpus_module = _load_compile_corpus_module()
    _validate_corpus_metadata(corpus_module)
    reference_torch = getattr(corpus_module, "reference_torch", None)
    if reference_torch is None:
        raise EvaluationFatalError("reference PyTorch is not importable")
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise EvaluationFatalError(
            "torch.compile coverage evaluation requires pinned PyTorch "
            f"{REFERENCE_PYTORCH_VERSION}, got {reference_torch.__version__}"
        )

    cases = _select_subset(corpus_module, subset)
    scenarios = _select_guard_subset(corpus_module, subset)
    reference_case_results, reference_scenario_results = _reference_results(
        corpus_module,
        reference_torch,
        cases,
        scenarios,
    )
    worker_payload = _run_candidate_subprocess(subset)
    verdicts, _candidate_scenarios = _compare_worker_to_reference(
        corpus_module,
        cases,
        reference_case_results,
        reference_scenario_results,
        worker_payload,
    )
    score, category_scores = _score_case_verdicts(
        corpus_module.CATEGORY_WEIGHTS,
        verdicts,
    )
    return _summarize_output(
        corpus_module,
        reference_torch,
        worker_payload,
        verdicts,
        category_scores,
        score,
        subset,
    )


def _fatal_output(error):
    return {
        "score": 0,
        "summary": f"torch.compile coverage evaluation failed closed: {error}",
        "evidence": [
            f"{EVALUATOR_VERSION} could not produce a trusted score.",
            f"Python executable: {sys.executable}",
            f"Python version: {sys.version.replace(chr(10), ' ')}",
            f"Platform: {platform.platform()}",
        ],
        "suggestions": [
            "Run from an environment with the current torch_rs build installed.",
            f"Install the reference dependency group so torch=={REFERENCE_PYTORCH_VERSION} is importable.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=("full", "public"),
        default="full",
        help="run the full corpus or the public strict subset",
    )
    parser.add_argument(
        "--candidate-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.candidate_worker:
        print(json.dumps(_run_candidate_worker(args), sort_keys=True))
        return 0
    try:
        output = evaluate(args.subset)
    except EvaluationFatalError as error:
        print(json.dumps(_fatal_output(error), indent=2, sort_keys=True))
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
