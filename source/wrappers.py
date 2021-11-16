import torch
import torch.nn as nn

class PredictRiskWrapper(nn.Module):
    def __init__(self, trained_model, timepoint, method):
        """
        :param trained_model: trained model to compute attribution for
        :param attribution_endpoint: endpoint to attribute to
        :param attribution_timepoint: timepoint to attribute to (currently not used for DeepSurv)
        :param method: ['KernelExplainer','DeepExplainer']
        """
        super().__init__()
        assert isinstance(timepoint, (int, float)), 'timepoint must be an instance of (int,float)'
        self.trained_model = trained_model
        self.timepoint = timepoint
        self.method = method
        self.bhs = trained_model.bhs

    def forward(self, inputs, t=None):
        t = self.timepoint
        inputs = inputs if self.method == 'DeepExplainer' else torch.Tensor(inputs)
        log_ph = self.trained_model(inputs, t=self.timepoint)[0]
        t = t.item()

        try:
            # Check if t is in self.bhs
            cum_bhs = self.bhs.at[t, 'cum_bhs']
        except KeyError:
            # If t is not in dataframe, interpolate and insert t to the dataframe
            t_row = pd.Series({'bhs': float('nan'), 'cum_bhs': float('nan')}, name=t)
            self.bhs = self.bhs.append(t_row).sort_index().interpolate(method='linear', axis=0)
            cum_bhs = self.bhs.at[t, 'cum_bhs']

        S_t = torch.exp(-torch.exp(log_ph) * cum_bhs)
        return 1 - S_t

class ForwardWrapper(nn.Module):
    def __init__(self, trained_model, method, **kwargs):
        """
        :param trained_model: trained model to compute attribution for
        :param attribution_endpoint: endpoint to attribute to
        :param attribution_timepoint: timepoint to attribute to (currently not used for DeepSurv)
        :param method: ['KernelExplainer','DeepExplainer']
        """
        super().__init__()
        self.trained_model = trained_model
        self.method = method
        self.bhs = trained_model.bhs

    def forward(self, inputs, t=None):
        inputs = inputs if self.method == 'DeepExplainer' else torch.Tensor(inputs)
        return self.trained_model(inputs)[0]

