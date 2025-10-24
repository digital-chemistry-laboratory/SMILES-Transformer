import os

from torch import nn
import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertForSequenceClassification, BertModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Tuple


class FeatureAugmentedClassificationBERT(BertForSequenceClassification):
    """
    BERT model for sequence classification augmented with additional features.

    This class extends `BertForSequenceClassification` by incorporating additional features into
    the classification layers. The additional features are concatenated with the pooled output from
    BERT before being passed through the classifier.

    Attributes:
        num_labels (int): Number of labels for classification.
        config (PretrainedConfig): Configuration object.
        bert (BertModel): The BERT model.
        dropout (nn.Dropout): Dropout layer applied to the pooled output.
        classifier (nn.Sequential): Classification head consisting of linear layers, activations, and dropout.
        n_features (int): Number of additional features to include.

    Methods:
        forward(...): Forward pass of the model.
        from_pretrained_downstream(...): Class method to load a pretrained model with additional features.
    """

    def __init__(self, config):
        """
        Initialize the FeatureAugmentedClassificationBERT model.

        Args:
            config (PretrainedConfig): Configuration object with model parameters.

        The initialization sets up the BERT model, the classifier layers, and applies any specified
        dropout rates. The classifier layers are constructed to accommodate additional features,
        which are concatenated with the BERT pooled output before classification.

        Note:
            The number of additional features (`n_features`) must be set on the class before initialization.
            This can be done by calling the `from_pretrained_downstream` method.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # Here we define the FFN on top:
        layers = [
            nn.LayerNorm(
                self.config.hidden_size + self.n_features,
                eps=self.config.layer_norm_eps,
            ),
            nn.Linear(
                in_features=config.hidden_size + self.n_features,
                out_features=self.layer_width,
            ),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
        ]
        for i in range(self.n_layers):
            layers.append(
                nn.Linear(in_features=self.layer_width, out_features=self.layer_width)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(classifier_dropout))
        layers.append(
            nn.Linear(in_features=self.layer_width, out_features=self.num_labels)
        )
        if self.num_labels > 1:
            layers.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*layers)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        additional_features: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        Perform a forward pass of the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.

            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.

            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs.

            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.

            additional_features (`torch.FloatTensor` of shape `(batch_size, n_features)`, *optional*):
                Additional features to be concatenated with the pooled output.

            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.

            labels (`torch.LongTensor` of shape `(batch_size,)` or `(batch_size, num_labels)`, *optional*):
                Labels for computing the loss.

            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers.

            output_hidden_states (`bool`, *optional*):
                Whether to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `Union[Tuple[torch.Tensor], SequenceClassifierOutput]`:
                If `return_dict=False`, returns a tuple containing:
                    - logits (`torch.Tensor`): Classification scores before SoftMax.
                    - hidden_states (optional): Hidden states of all layers.
                    - attentions (optional): Attention weights of all layers.
                If `return_dict=True`, returns a `SequenceClassifierOutput` containing:
                    - loss (`torch.Tensor`, optional): Classification (or regression) loss.
                    - logits (`torch.Tensor`): Classification scores before SoftMax.
                    - hidden_states (optional): Hidden states of all layers.
                    - attentions (optional): Attention weights of all layers.

        Notes:
            The additional features are concatenated with the pooled output from BERT before being passed through the classifier.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        concat_layer = torch.concat([pooled_output, additional_features], -1)
        logits = self.classifier(concat_layer)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained_downstream(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        num_labels: int,
        n_features: int,
        n_layers: int,
        layer_width: int,
    ):
        """
        Load a pretrained model for downstream tasks with additional features.

        This method sets the number of additional features and loads a pretrained model.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Path to pretrained model or model identifier from huggingface.co/models.

            num_labels (`int`):
                Number of labels for classification.

            n_features (`int`):
                Number of additional features to include.

        Returns:
            `FeatureAugmentedClassificationBERT`: An instance of the model with pretrained weights.
        """
        cls.n_features = n_features
        cls.n_layers = n_layers
        cls.layer_width = layer_width
        return cls.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
        )
