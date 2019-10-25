
# from https://github.com/allenai/allennlp/blob/master/allennlp/models/model.py

"""
:py:class:`Model` is an abstract class representing
an AllenNLP model.
"""

import logging
import os
from typing import Dict, Union, List, Set

import numpy
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.nn import util
from allennlp.nn.regularizers import RegularizerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# When training a model, many sets of weights are saved. By default we want to
# save/load this set of weights.
_DEFAULT_WEIGHTS = "best.th"


class Model(torch.nn.Module, Registrable):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of ``forward`` to be a dictionary.
    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors.
    In order for your model to be trained using the :class:`~allennlp.training.Trainer`
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.
    Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
    of early stopping and best-model serialization based on a validation metric in
    :class:`~allennlp.training.Trainer`. Metrics that begin with "_" will not be logged
    to the progress bar by :class:`~allennlp.training.Trainer`.
    """
    _warn_for_unseparable_batches: Set[str] = set()

    def __init__(self,
                 vocab: Vocabulary,
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__()
        self.vocab = vocab
        self._regularizer = regularizer

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:
        """
        Computes the regularization penalty for the model.
        Returns 0 if the model was not configured to use regularization.
        """
        if self._regularizer is None:
            return 0.0
        else:
            return self._regularizer(self)

    def get_parameters_for_histogram_tensorboard_logging( # pylint: disable=invalid-name
            self) -> List[str]:
        """
        Returns the name of model parameters used for logging histograms to tensorboard.
        """
        return [name for name, _ in self.named_parameters()]

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:  # pylint: disable=arguments-differ
        """
        Defines the forward pass of the model. In addition, to facilitate easy training,
        this method is designed to compute a loss function defined by a user.
        The input is comprised of everything required to perform a
        training update, `including` labels - you define the signature here!
        It is down to the user to ensure that inference can be performed
        without the presence of these labels. Hence, any inputs not available at
        inference time should only be used inside a conditional block.
        The intended sketch of this method is as follows::
            def forward(self, input1, input2, targets=None):
                ....
                ....
                output1 = self.layer1(input1)
                output2 = self.layer2(input2)
                output_dict = {"output1": output1, "output2": output2}
                if targets is not None:
                    # Function returning a scalar torch.Tensor, defined by the user.
                    loss = self._compute_loss(output1, output2, targets)
                    output_dict["loss"] = loss
                return output_dict
        Parameters
        ----------
        inputs:
            Tensors comprising everything needed to perform a training update, `including` labels,
            which should be optional (i.e have a default value of ``None``).  At inference time,
            simply pass the relevant inputs, not including the labels.
        Returns
        -------
        output_dict: ``Dict[str, torch.Tensor]``
            The outputs from the model. In order to train a model using the
            :class:`~allennlp.training.Trainer` api, you must provide a "loss" key pointing to a
            scalar ``torch.Tensor`` representing the loss to be optimized.
        """
        raise NotImplementedError

    def forward_on_instance(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        Takes an :class:`~allennlp.data.instance.Instance`, which typically has raw text in it,
        converts that text into arrays using this model's :class:`Vocabulary`, passes those arrays
        through :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and remove the batch dimension.
        """
        return self.forward_on_instances([instance])[0]

    def forward_on_instances(self,
                             instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        """
        Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
        arrays using this model's :class:`Vocabulary`, passes those arrays through
        :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and separate the
        batched output into a list of individual dicts per instance. Note that typically
        this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
        :func:`forward_on_instance`.
        Parameters
        ----------
        instances : List[Instance], required
            The instances to run the model on.
        cuda_device : int, required
            The GPU device to use.  -1 means use the CPU.
        Returns
        -------
        A list of the models output for each instance.
        """
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.decode(self(**model_input))

            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                outputs[name] = output
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Takes the result of :func:`forward` and runs inference / decoding / whatever
        post-processing you need to do your model.  The intent is that ``model.forward()`` should
        produce potentials or probabilities, and then ``model.decode()`` can take those results and
        run some kind of beam search or constrained inference or whatever is necessary.  This does
        not handle all possible decoding use cases, but it at least handles simple kinds of
        decoding.
        This method `modifies` the input dictionary, and also `returns` the same dictionary.
        By default in the base class we do nothing.  If your model has some special decoding step,
        override this method.
        """
        # pylint: disable=no-self-use
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Returns a dictionary of metrics. This method will be called by
        :class:`allennlp.training.Trainer` in order to compute and use model metrics for early
        stopping and model serialization.  We return an empty dictionary here rather than raising
        as it is not required to implement metrics for a new model.  A boolean `reset` parameter is
        passed, as frequently a metric accumulator will have some state which should be reset
        between epochs. This is also compatible with :class:`~allennlp.training.Metric`s. Metrics
        should be populated during the call to ``forward``, with the
        :class:`~allennlp.training.Metric` handling the accumulation of the metric until this
        method is called.
        """
        # pylint: disable=unused-argument,no-self-use
        return {}

    def _get_prediction_device(self) -> int:
        """
        This method checks the device of the model parameters to determine the cuda_device
        this model should be run on for predictions.  If there are no parameters, it returns -1.
        Returns
        -------
        The cuda device this model should run on for predictions.
        """
        devices = {util.get_device_of(param) for param in self.parameters()}

        if len(devices) > 1:
            devices_string = ", ".join(str(x) for x in devices)
            raise ConfigurationError(f"Parameters have mismatching cuda_devices: {devices_string}")
        elif len(devices) == 1:
            return devices.pop()
        else:
            return -1

    def _maybe_warn_for_unseparable_batches(self, output_key: str):
        """
        This method warns once if a user implements a model which returns a dictionary with
        values which we are unable to split back up into elements of the batch. This is controlled
        by a class attribute ``_warn_for_unseperable_batches`` because it would be extremely verbose
        otherwise.
        """
        if  output_key not in self._warn_for_unseparable_batches:
            logger.warning(f"Encountered the {output_key} key in the model's return dictionary which "
                           "couldn't be split by the batch size. Key will be ignored.")
            # We only want to warn once for this key,
            # so we set this to false so we don't warn again.
            self._warn_for_unseparable_batches.add(output_key)

    @classmethod
    def _load(cls,
              config: Params,
              serialization_dir: str,
              weights_file: str = None,
              cuda_device: int = -1) -> 'Model':
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, 'vocabulary')
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
        vocab = Vocabulary.by_name(vocab_choice).from_files(vocab_dir)

        model_params = config.get('model')

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        remove_pretrained_embedding_params(model_params)
        model = Model.from_params(vocab=vocab, params=model_params)
        model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))

        # TERRIBLE WORKAROUND, DERIVED FROM ERROR MESSAGES
        missing_keys = ["_states.xintent.embedder.weight", "_states.xintent.decoder_cell.weight_ih", "_states.xintent.decoder_cell.weight_hh", "_states.xintent.decoder_cell.bias_ih", "_states.xintent.decoder_cell.bias_hh", "_states.xintent.output_projection_layer.weight", "_states.xintent.output_projection_layer.bias", "_states.xreact.embedder.weight", "_states.xreact.decoder_cell.weight_ih", "_states.xreact.decoder_cell.weight_hh", "_states.xreact.decoder_cell.bias_ih", "_states.xreact.decoder_cell.bias_hh", "_states.xreact.output_projection_layer.weight", "_states.xreact.output_projection_layer.bias", "_states.oreact.embedder.weight", "_states.oreact.decoder_cell.weight_ih", "_states.oreact.decoder_cell.weight_hh", "_states.oreact.decoder_cell.bias_ih", "_states.oreact.decoder_cell.bias_hh", "_states.oreact.output_projection_layer.weight", "_states.oreact.output_projection_layer.bias"]
        unexpected_keys = ["xintent_embedder.weight", "xintent_decoder_cell.weight_ih", "xintent_decoder_cell.weight_hh", "xintent_decoder_cell.bias_ih", "xintent_decoder_cell.bias_hh", "xintent_output_project_layer.weight", "xintent_output_project_layer.bias", "xreact_embedder.weight", "xreact_decoder_cell.weight_ih", "xreact_decoder_cell.weight_hh", "xreact_decoder_cell.bias_ih", "xreact_decoder_cell.bias_hh", "xreact_output_project_layer.weight", "xreact_output_project_layer.bias", "oreact_embedder.weight", "oreact_decoder_cell.weight_ih", "oreact_decoder_cell.weight_hh", "oreact_decoder_cell.bias_ih", "oreact_decoder_cell.bias_hh", "oreact_output_project_layer.weight", "oreact_output_project_layer.bias"]
        for missing_key, unexpected_key in zip(missing_keys, unexpected_keys):
            model_state[missing_key] = model_state[unexpected_key]
            del model_state[unexpected_key]

        model.load_state_dict(model_state)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        return model

    @classmethod
    def load(cls,
             config: Params,
             serialization_dir: str,
             weights_file: str = None,
             cuda_device: int = -1) -> 'Model':
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        Parameters
        ----------
        config: Params
            The configuration that was used to train the model. It should definitely
            have a `model` section, and should probably have a `trainer` section
            as well.
        serialization_dir: str = None
            The directory containing the serialized weights, parameters, and vocabulary
            of the model.
        weights_file: str = None
            By default we load the weights from `best.th` in the serialization
            directory, but you can override that value here.
        cuda_device: int = -1
            By default we load the model on the CPU, but if you want to load it
            for GPU usage you can specify the id of your GPU here
        Returns
        -------
        model: Model
            The model specified in the configuration, loaded with the serialized
            vocabulary and the trained weights.
        """

        # Peak at the class of the model.
        model_type = config["model"]["type"]

        # Load using an overridable _load method.
        # This allows subclasses of Model to override _load.
        # pylint: disable=protected-access
        return cls.by_name(model_type)._load(config, serialization_dir, weights_file, cuda_device)


def remove_pretrained_embedding_params(params: Params):
    keys = params.keys()
    if 'pretrained_file' in keys:
        del params['pretrained_file']
    for value in params.values():
        if isinstance(value, Params):
            remove_pretrained_embedding_params(value)

################################################################################
# from https://github.com/allenai/allennlp/blob/master/allennlp/models/event2mind.py

from typing import Dict, List, Optional, Tuple

import numpy
from overrides import overrides

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.rnn import GRUCell
from torch.nn.modules.linear import Linear
from torch import nn
import torch.nn.functional as F

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
# from allennlp.models.model import Model
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import UnigramRecall


@Model.register("event2mind")
class Event2Mind(Model):
    """
    This ``Event2Mind`` class is a :class:`Model` which takes an event
    sequence, encodes it, and then uses the encoded representation to decode
    several mental state sequences.
    It is based on `the paper by Rashkin et al.
    <https://www.semanticscholar.org/paper/Event2Mind/b89f8a9b2192a8f2018eead6b135ed30a1f2144d>`_
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (``tokens``) or the target tokens can have a different namespace, in which case it needs to
        be specified as ``target_namespace``.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences.
    embedding_dropout: float, required
        The amount of dropout to apply after the source tokens have been embedded.
    encoder : ``Seq2VecEncoder``, required
        The encoder of the "encoder/decoder" model.
    max_decoding_steps : int, required
        Length of decoded sequences.
    beam_size : int, optional (default = 10)
        The width of the beam search.
    target_names: ``List[str]``, optional, (default = ['xintent', 'xreact', 'oreact'])
        Names of the target fields matching those in the ``Instance`` objects.
    target_namespace : str, optional (default = 'tokens')
        If the target side vocabulary is different from the source side's, you need to specify the
        target's namespace here. If not, we'll assume it is "tokens", which is also the default
        choice for the source side, and this might cause them to share vocabularies.
    target_embedding_dim : int, optional (default = source_embedding_dim)
        You can specify an embedding dimensionality for the target side. If not, we'll use the same
        value as the source embedder's.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 encoder: Seq2VecEncoder,
                 max_decoding_steps: int,
                 beam_size: int = 10,
                 target_names: List[str] = None,
                 target_namespace: str = "tokens",
                 target_embedding_dim: int = None) -> None:
        super().__init__(vocab)
        target_names = target_names or ["xintent", "xreact", "oreact"]

        # Note: The original tweaks the embeddings for "personx" to be the mean
        # across the embeddings for "he", "she", "him" and "her". Similarly for
        # "personx's" and so forth. We could consider that here as a well.
        self._source_embedder = source_embedder
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._encoder = encoder
        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        # Warning: The different decoders share a vocabulary! This may be
        # counterintuitive, but consider the case of xreact and oreact. A
        # reaction of "happy" could easily apply to both the subject of the
        # event and others. This could become less appropriate as more decoders
        # are added.
        num_classes = self.vocab.get_vocab_size(self._target_namespace)
        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder.
        self._decoder_output_dim = self._encoder.get_output_dim()
        target_embedding_dim = target_embedding_dim or self._source_embedder.get_output_dim()

        self._states = ModuleDict()
        for name in target_names:
            self._states[name] = StateDecoder(
                    num_classes,
                    target_embedding_dim,
                    self._decoder_output_dim
            )

        self._beam_search = BeamSearch(
                self._end_index,
                beam_size=beam_size,
                max_steps=max_decoding_steps
        )

    def _update_recall(self,
                       all_top_k_predictions: torch.Tensor,
                       target_tokens: Dict[str, torch.LongTensor],
                       target_recall: UnigramRecall) -> None:
        targets = target_tokens["tokens"]
        target_mask = get_text_field_mask(target_tokens)
        # See comment in _get_loss.
        # TODO(brendanr): Do we need contiguous here?
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()
        target_recall(
                all_top_k_predictions,
                relevant_targets,
                relevant_mask,
                self._end_index
        )

    def _get_num_decoding_steps(self,
                                target_tokens: Optional[Dict[str, torch.LongTensor]]) -> int:
        if target_tokens:
            targets = target_tokens["tokens"]
            target_sequence_length = targets.size()[1]
            # The last input from the target is either padding or the end
            # symbol.  Either way, we don't have to process it. (To be clear,
            # we do still output and compare against the end symbol, but there
            # is no need to take the end symbol as input to the decoder.)
            return target_sequence_length - 1
        else:
            return self._max_decoding_steps

    @overrides
    def forward(self,  # type: ignore
                source: Dict[str, torch.LongTensor],
                **target_tokens: Dict[str, Dict[str, torch.LongTensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the target sequences.
        Parameters
        ----------
        source : ``Dict[str, torch.LongTensor]``
            The output of ``TextField.as_array()`` applied on the source
            ``TextField``. This will be passed through a ``TextFieldEmbedder``
            and then through an encoder.
        target_tokens : ``Dict[str, Dict[str, torch.LongTensor]]``:
            Dictionary from name to output of ``Textfield.as_array()`` applied
            on target ``TextField``. We assume that the target tokens are also
            represented as a ``TextField``.
        """
        # (batch_size, input_sequence_length, embedding_dim)
        embedded_input = self._embedding_dropout(self._source_embedder(source))
        source_mask = get_text_field_mask(source)
        # (batch_size, encoder_output_dim)
        final_encoder_output = self._encoder(embedded_input, source_mask)
        output_dict = {}

        # Perform greedy search so we can get the loss.
        if target_tokens:
            if target_tokens.keys() != self._states.keys():
                target_only = target_tokens.keys() - self._states.keys()
                states_only = self._states.keys() - target_tokens.keys()
                raise Exception("Mismatch between target_tokens and self._states. Keys in " +
                                f"targets only: {target_only} Keys in states only: {states_only}")
            total_loss = 0
            for name, state in self._states.items():
                loss = self.greedy_search(
                        final_encoder_output=final_encoder_output,
                        target_tokens=target_tokens[name],
                        target_embedder=state.embedder,
                        decoder_cell=state.decoder_cell,
                        output_projection_layer=state.output_projection_layer
                )
                total_loss += loss
                output_dict[f"{name}_loss"] = loss

            # Use mean loss (instead of the sum of the losses) to be comparable to the paper.
            output_dict["loss"] = total_loss / len(self._states)

        # Perform beam search to obtain the predictions.
        if not self.training:
            batch_size = final_encoder_output.size()[0]
            for name, state in self._states.items():
                start_predictions = final_encoder_output.new_full(
                        (batch_size,), fill_value=self._start_index, dtype=torch.long)
                start_state = {"decoder_hidden": final_encoder_output}

                # (batch_size, 10, num_decoding_steps)
                all_top_k_predictions, log_probabilities = self._beam_search.search(
                        start_predictions, start_state, state.take_step)

                if target_tokens:
                    self._update_recall(all_top_k_predictions, target_tokens[name], state.recall)
                output_dict[f"{name}_top_k_predictions"] = all_top_k_predictions
                output_dict[f"{name}_top_k_log_probabilities"] = log_probabilities

        return output_dict

    def greedy_search(self,
                      final_encoder_output: torch.LongTensor,
                      target_tokens: Dict[str, torch.LongTensor],
                      target_embedder: Embedding,
                      decoder_cell: GRUCell,
                      output_projection_layer: Linear) -> torch.FloatTensor:
        """
        Greedily produces a sequence using the provided ``decoder_cell``.
        Returns the cross entropy between this sequence and ``target_tokens``.
        Parameters
        ----------
        final_encoder_output : ``torch.LongTensor``, required
            Vector produced by ``self._encoder``.
        target_tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()`` applied on some target ``TextField``.
        target_embedder : ``Embedding``, required
            Used to embed the target tokens.
        decoder_cell: ``GRUCell``, required
            The recurrent cell used at each time step.
        output_projection_layer: ``Linear``, required
            Linear layer mapping to the desired number of classes.
        """
        num_decoding_steps = self._get_num_decoding_steps(target_tokens)
        targets = target_tokens["tokens"]
        decoder_hidden = final_encoder_output
        step_logits = []
        for timestep in range(num_decoding_steps):
            # See https://github.com/allenai/allennlp/issues/1134.
            input_choices = targets[:, timestep]
            decoder_input = target_embedder(input_choices)
            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size, num_classes)
            output_projections = output_projection_layer(decoder_hidden)
            # list of (batch_size, 1, num_classes)
            step_logits.append(output_projections.unsqueeze(1))
        # (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)
        target_mask = get_text_field_mask(target_tokens)
        return self._get_loss(logits, targets, target_mask)

    def greedy_predict(self,
                       final_encoder_output: torch.LongTensor,
                       target_embedder: Embedding,
                       decoder_cell: GRUCell,
                       output_projection_layer: Linear) -> torch.Tensor:
        """
        Greedily produces a sequence using the provided ``decoder_cell``.
        Returns the predicted sequence.
        Parameters
        ----------
        final_encoder_output : ``torch.LongTensor``, required
            Vector produced by ``self._encoder``.
        target_embedder : ``Embedding``, required
            Used to embed the target tokens.
        decoder_cell: ``GRUCell``, required
            The recurrent cell used at each time step.
        output_projection_layer: ``Linear``, required
            Linear layer mapping to the desired number of classes.
        """
        num_decoding_steps = self._max_decoding_steps
        decoder_hidden = final_encoder_output
        batch_size = final_encoder_output.size()[0]
        predictions = [final_encoder_output.new_full(
                (batch_size,), fill_value=self._start_index, dtype=torch.long
        )]
        for _ in range(num_decoding_steps):
            input_choices = predictions[-1]
            decoder_input = target_embedder(input_choices)
            decoder_hidden = decoder_cell(decoder_input, decoder_hidden)
            # (batch_size, num_classes)
            output_projections = output_projection_layer(decoder_hidden)
            class_probabilities = F.softmax(output_projections, dim=-1)
            _, predicted_classes = torch.max(class_probabilities, 1)
            predictions.append(predicted_classes)
        all_predictions = torch.cat([ps.unsqueeze(1) for ps in predictions], 1)
        # Drop start symbol and return.
        return all_predictions[:, 1:]

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.FloatTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.
        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss

    def decode_all(self, predicted_indices: torch.Tensor) -> List[List[str]]:
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_tokens = []
        for indices in predicted_indices:
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[:indices.index(self._end_index)]
            predicted_tokens = [self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                                for x in indices]
            all_predicted_tokens.append(predicted_tokens)
        return all_predicted_tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, List[List[str]]]:
        """
        This method overrides ``Model.decode``, which gets called after ``Model.forward``, at test
        time, to finalize predictions. The logic for the decoder part of the encoder-decoder lives
        within the ``forward`` method.
        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds fields for the tokens to the ``output_dict``.
        """
        for name in self._states:
            top_k_predicted_indices = output_dict[f"{name}_top_k_predictions"][0]
            output_dict[f"{name}_top_k_predicted_tokens"] = [self.decode_all(top_k_predicted_indices)]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics = {}
        # Recall@10 needs beam search which doesn't happen during training.
        if not self.training:
            for name, state in self._states.items():
                all_metrics[name] = state.recall.get_metric(reset=reset)
        return all_metrics


class StateDecoder(Module):
    # pylint: disable=abstract-method
    """
    Simple struct-like class for internal use.
    """
    def __init__(self,
                 num_classes: int,
                 input_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self.embedder = Embedding(num_classes, input_dim)
        self.decoder_cell = GRUCell(input_dim, output_dim)
        self.output_projection_layer = Linear(output_dim, num_classes)
        self.recall = UnigramRecall()

    def take_step(self,
                  last_predictions: torch.Tensor,
                  state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        decoder_hidden = state["decoder_hidden"]
        decoder_input = self.embedder(last_predictions)
        decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
        state["decoder_hidden"] = decoder_hidden
        output_projections = self.output_projection_layer(decoder_hidden)
        class_log_probabilities = F.log_softmax(output_projections, dim=-1)
        return class_log_probabilities, state

################################################################################
# from https://github.com/allenai/allennlp/blob/master/allennlp/models/archival.py:

"""
Helper functions for archiving models and restoring archived models.
"""

from typing import NamedTuple, Dict, Any
import json
import logging
import os
import tempfile
import tarfile
import shutil

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params, unflatten, with_fallback, parse_overrides
from allennlp.models.model import _DEFAULT_WEIGHTS # Model, _DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# An archive comprises a Model and its experimental config
Archive = NamedTuple("Archive", [("model", Model), ("config", Params)])

# We archive a model by creating a tar.gz file with its weights, config, and vocabulary.
#
# We also may include other arbitrary files in the archive. In this case we store
# the mapping { flattened_path -> filename } in ``files_to_archive.json`` and the files
# themselves under the path ``fta/`` .
#
# These constants are the *known names* under which we archive them.
CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "weights.th"
_FTA_NAME = "files_to_archive.json"

def archive_model(serialization_dir: str,
                  weights: str = _DEFAULT_WEIGHTS,
                  files_to_archive: Dict[str, str] = None) -> None:
    """
    Archive the model weights, its training configuration, and its
    vocabulary to `model.tar.gz`. Include the additional ``files_to_archive``
    if provided.
    Parameters
    ----------
    serialization_dir: ``str``
        The directory where the weights and vocabulary are written out.
    weights: ``str``, optional (default=_DEFAULT_WEIGHTS)
        Which weights file to include in the archive. The default is ``best.th``.
    files_to_archive: ``Dict[str, str]``, optional (default=None)
        A mapping {flattened_key -> filename} of supplementary files to include
        in the archive. That is, if you wanted to include ``params['model']['weights']``
        then you would specify the key as `"model.weights"`.
    """
    weights_file = os.path.join(serialization_dir, weights)
    if not os.path.exists(weights_file):
        logger.error("weights file %s does not exist, unable to archive model", weights_file)
        return

    config_file = os.path.join(serialization_dir, CONFIG_NAME)
    if not os.path.exists(config_file):
        logger.error("config file %s does not exist, unable to archive model", config_file)

    # If there are files we want to archive, write out the mapping
    # so that we can use it during de-archiving.
    if files_to_archive:
        fta_filename = os.path.join(serialization_dir, _FTA_NAME)
        with open(fta_filename, 'w') as fta_file:
            fta_file.write(json.dumps(files_to_archive))


    archive_file = os.path.join(serialization_dir, "model.tar.gz")
    logger.info("archiving weights and vocabulary to %s", archive_file)
    with tarfile.open(archive_file, 'w:gz') as archive:
        archive.add(config_file, arcname=CONFIG_NAME)
        archive.add(weights_file, arcname=_WEIGHTS_NAME)
        archive.add(os.path.join(serialization_dir, "vocabulary"),
                    arcname="vocabulary")

        # If there are supplemental files to archive:
        if files_to_archive:
            # Archive the { flattened_key -> original_filename } mapping.
            archive.add(fta_filename, arcname=_FTA_NAME)
            # And add each requested file to the archive.
            for key, filename in files_to_archive.items():
                archive.add(filename, arcname=f"fta/{key}")

def load_event2mind_archive(archive_file: str,
                 cuda_device: int = -1,
                 overrides: str = "",
                 weights_file: str = None) -> Archive:
    """
    Instantiates an Archive from an archived `tar.gz` file.
    Parameters
    ----------
    archive_file: ``str``
        The archive file to load the model from.
    weights_file: ``str``, optional (default = None)
        The weights file to use.  If unspecified, weights.th in the archive_file will be used.
    cuda_device: ``int``, optional (default = -1)
        If `cuda_device` is >= 0, the model will be loaded onto the
        corresponding GPU. Otherwise it will be loaded onto the CPU.
    overrides: ``str``, optional (default = "")
        JSON overrides to apply to the unarchived ``Params`` object.
    """
    # redirect to the cache, if necessary
    resolved_archive_file = cached_path(archive_file)

    if resolved_archive_file == archive_file:
        logger.info(f"loading archive file {archive_file}")
    else:
        logger.info(f"loading archive file {archive_file} from cache at {resolved_archive_file}")

    tempdir = None
    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info(f"extracting archive file {resolved_archive_file} to temp dir {tempdir}")
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            archive.extractall(tempdir)

        serialization_dir = tempdir

    # Check for supplemental files in archive
    fta_filename = os.path.join(serialization_dir, _FTA_NAME)
    if os.path.exists(fta_filename):
        with open(fta_filename, 'r') as fta_file:
            files_to_archive = json.loads(fta_file.read())

        # Add these replacements to overrides
        replacements_dict: Dict[str, Any] = {}
        for key, _ in files_to_archive.items():
            replacement_filename = os.path.join(serialization_dir, f"fta/{key}")
            replacements_dict[key] = replacement_filename

        overrides_dict = parse_overrides(overrides)
        combined_dict = with_fallback(preferred=unflatten(replacements_dict), fallback=overrides_dict)
        overrides = json.dumps(combined_dict)

    # Load config
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)
    config.loading_from_archive = True

    if weights_file:
        weights_path = weights_file
    else:
        weights_path = os.path.join(serialization_dir, _WEIGHTS_NAME)

    # Instantiate model. Use a duplicate of the config, as it will get consumed.
    model = Model.load(config.duplicate(),
                       weights_file=weights_path,
                       serialization_dir=serialization_dir,
                       cuda_device=cuda_device)

    if tempdir:
        # Clean up temp dir
        shutil.rmtree(tempdir)

    return Archive(model=model, config=config)
