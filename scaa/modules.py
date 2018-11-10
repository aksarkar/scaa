import itertools
import scaa
import torch

class Encoder(torch.nn.Module):
  """Encoder q(z | x) = N(mu(x), sigma^2(x) I)

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
    # scVI does not play nicely
    with torch.set_grad_enabled(True):
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

  def loss(self, x, stoch_samples):
    mean, scale = self.encoder.forward(x)
    assert mean.requires_grad
    assert scale.requires_grad
    # [batch_size]
    # Important: this is analytic
    kl_term = torch.sum(scaa.loss.kl_term(mean, scale), dim=1)
    # [stoch_samples, batch_size, latent_dim]
    qz = torch.distributions.Normal(mean, scale).rsample(stoch_samples)
    # [stoch_samples, batch_size, input_dim]
    logodds, mean = self.decoder.forward(qz)
    assert logodds.requires_grad
    assert mean.requires_grad
    error_term = torch.mean(torch.sum(scaa.loss.zip_llik(x, mean, logodds), dim=2), dim=0)
    # Important: optim minimizes
    loss = -torch.sum(error_term - kl_term)
    return loss

  def fit(self, x, max_epochs, verbose=False, stoch_samples=10, **kwargs):
    """Fit the model

    :param x: torch.utils.data.DataLoader

    """
    stoch_samples = torch.Size([stoch_samples])
    opt = torch.optim.Adam(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      for i, batch in enumerate(x):
        opt.zero_grad()
        loss = self.loss(batch, stoch_samples)
        loss.backward()
        opt.step()
        if verbose and not i % 10:
          print(f'[epoch={epoch} batch={i}] elbo={-loss}')
    return self

  @torch.no_grad()
  def denoise(self, x):
    # Plug E[z | x] into the decoder
    return torch.cat([self.decoder.forward(self.encoder.forward(batch)[0])[1] for batch in x]).numpy()

class Discriminator(torch.nn.Module):
  def __init__(self, input_dim, num_classes):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, input_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(input_dim, input_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(input_dim, num_classes),
      torch.nn.LogSoftmax(dim=1),
    )

  def forward(self, x):
    return self.net(x)

class ZIPAAE(torch.nn.Module):
  def __init__(self, input_dim, latent_dim, num_classes):
    super().__init__()
    self.vae = ZIPVAE(input_dim, latent_dim)
    self.adv = Discriminator(latent_dim, num_classes)

  def fit(self, x, y, max_epochs, verbose=False, stoch_samples=10, **kwargs):
    """Fit the model

    :x: torch.utils.data.DataLoader
    :y: torch.utils.data.DataLoader

    """
    stoch_samples = torch.Size([stoch_samples])
    # Reconstruction
    opt0 = torch.optim.Adam(self.vae.parameters(), **kwargs)
    # Adversary
    opt1 = torch.optim.Adam(self.adv.parameters(), **kwargs)
    # Generator
    opt2 = torch.optim.Adam(self.vae.encoder.parameters(), **kwargs)
    for epoch in range(max_epochs):
      for i, (batch_x, batch_y) in enumerate(zip(x, y)):
        opt0.zero_grad()
        loss0 = self.vae.loss(batch_x, stoch_samples)
        loss0.backward()
        opt0.step()

        f = torch.nn.functional.cross_entropy

        # Fix the generator, train the adversary
        opt1.zero_grad()
        loss1 = f(self.adv.forward(self.vae.encoder.forward(batch_x)[0]), batch_y)
        loss1.backward()
        opt1.step()

        # Fix the adversary, train the generator
        opt2.zero_grad()
        loss2 = -f(self.adv.forward(self.vae.encoder.forward(batch_x)[0]), batch_y)
        loss2.backward()
        opt2.step()

        if verbose and not i % 10:
          print(f'[epoch={epoch} batch={i}] vae={loss0} adv={loss1} gen={loss2}')
    return self

  def denoise(self, x):
    return self.vae.denoise(x)

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
