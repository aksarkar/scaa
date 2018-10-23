import scaa
import torch

class Encoder(torch.nn.Module):
  """Encoder q(z | x) = N(mu(x), sigma^2(x))

  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
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
    return self.mean(q), self.scale(q)

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

  def fit(self, x, max_epochs, verbose=False, stoch_samples=10, **kwargs):
    """Fit the model

    :param x: torch.utils.data.DataLoader

    """
    stoch_samples = torch.Size([stoch_samples])
    opt = torch.optim.Adam(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      for i, batch in enumerate(x):
        opt.zero_grad()
        mean, scale = self.encoder.forward(batch)
        # [batch_size]
        # Important: this is analytic
        kl_term = torch.sum(scaa.loss.kl_term(mean, scale), dim=1)
        # [stoch_samples, batch_size, latent_dim]
        qz = torch.distributions.Normal(mean, scale).rsample(stoch_samples)
        # [stoch_samples, batch_size, input_dim]
        logodds, mean = self.decoder.forward(qz)
        error_term = torch.mean(torch.sum(scaa.loss.zip_llik(batch, mean, logodds), dim=2), dim=0)
        # Important: optim minimizes
        loss = -torch.sum(error_term - kl_term)
        loss.backward()
        opt.step()
        if verbose and not i % 10:
          print(f'[epoch={epoch} batch={i}] error={torch.sum(error_term)} kl={torch.sum(kl_term)} elbo={-loss}')
    return self

  def denoise(self, x):
    # Plug E[z | x] into the decoder
    return torch.cat([self.decoder.forward(self.encoder.forward(batch)[0])[1] for batch in x]).detach().numpy()

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
