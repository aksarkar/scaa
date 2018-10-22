import scaa
import torch

class Encoder(torch.nn.Module):
  """Encoder q(z | x) = N(mu(x), sigma^2(x))

  """
  def __init__(self, input_dim, output_dim, stoch_samples=10):
    super().__init__()
    self.stoch_samples = torch.Size([stoch_samples])
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
      torch.nn.ReLU(),
    )
    self.mean = torch.nn.Linear(128, output_dim)
    self.scale = torch.nn.Sequential(torch.nn.Linear(128, output_dim), torch.nn.Softplus())

  def forward(self, x):
    q = self.net(x)
    mean = self.mean(q)
    scale = self.scale(q)
    qz = torch.distributions.Normal(mean, scale).rsample(self.stoch_samples)
    return qz

class ZIP(torch.nn.Module):
  """Decoder p(x | z) = pi(z) delta0(.) + (1 - pi(z)) Poisson(lambda(z))

  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 128),
      torch.nn.ReLU(),
    )
    self.logodds = torch.nn.Linear(128, output_dim)
    self.rate = torch.nn.Sequential(torch.nn.Linear(128, output_dim), torch.nn.Softplus())

  def forward(self, x):
    p = self.net(x)
    return self.logodds(p), self.rate(p)

class ZIPVAE(torch.nn.Module):
  def __init__(self, input_dim, latent_dim):
    super().__init__()
    self.encoder = Encoder(input_dim, latent_dim)
    self.decoder = ZIP(latent_dim, input_dim)

  def fit(self, x, max_epochs, verbose=False, **kwargs):
    """Fit the model

    :param x: torch.utils.data.DataLoader

    """
    opt = torch.optim.Adam(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      for i, batch in enumerate(x):
        opt.zero_grad()
        # [stoch_samples, batch_size, latent_dim]
        qz = self.encoder.forward(batch)
        # [stoch_samples, batch_size, input_dim]
        logodds, mean = self.decoder.forward(qz)
        loss = torch.sum(torch.mean(torch.sum(scaa.loss.zip_llik(batch, mean, logodds), dim=2) - torch.sum(scaa.loss.kl_term(qz), dim=2), dim=0))
        loss.backward()
        opt.step()
        if verbose:
          print(f'{i} {loss}')

class BinaryDisciminator(torch.nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, input_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(input_dim, input_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(input_dim, 1),
      torch.nn.Sigmoid(),
    )

  def forward(self, x):
    return self.net(x)

class Generator(torch.nn.Module):
  """Generator G(z) = N(mu(z), sigma(z))

  """
  def __init__(self, p):
    self.net = torch.nn.Sequential(
      torch.nn.Linear(p, p),
      torch.nn.ReLU(),
      torch.nn.Linear(p, p),
      torch.nn.ReLU(),
    )
    self.mean = torch.nn.Linear(p, p)
    self.scale = torch.nn.Softplus(p, p)

  def forward(self, x):
    act = self.net(x)
    mean = self.mean(act)
    scale = self.scale(act)

class MulticlassDiscriminator(torch.nn.Module):
  def __init__(self):
    pass
