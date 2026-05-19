# Weighted Mean Derivation

Suppose we want to estimate an unknown true value μ from n independent observations.

Each observation is normally distributed around μ, but with its own variance:

X_i ~ N(μ, σ_i²)

We observe values x₁, x₂, ..., xₙ.

The variances σ₁², σ₂², ..., σₙ² represent the uncertainty in each observation.
Smaller variance means a more reliable observation, so we expect observations with smaller variance to contribute more heavily to our estimate of μ.

## Likelihood

For one observation:

p(x_i | μ, σ_i²)
= 1 / sqrt(2πσ_i²) * exp(-(x_i - μ)² / (2σ_i²))

Assuming independence, the joint likelihood is:

L(μ)
= Πᵢ 1 / sqrt(2πσ_i²) * exp(-(x_i - μ)² / (2σ_i²))

## Log-likelihood

Taking logs:

J(μ)
= Σᵢ log(1 / sqrt(2πσ_i²))
  - Σᵢ (x_i - μ)² / (2σ_i²)

The first term does not depend on μ, so the important part is:

- Σᵢ (x_i - μ)² / (2σ_i²)

## Differentiating

dJ/dμ = Σᵢ (x_i - μ) / σ_i²

Set this equal to zero:

0 = Σᵢ (x_i - μ) / σ_i²

0 = Σᵢ x_i / σ_i² - μ Σᵢ 1 / σ_i²

Therefore:

μ_hat = [Σᵢ x_i / σ_i²] / [Σᵢ 1 / σ_i²]

This is the inverse-variance weighted mean.

Each observation x_i is weighted by 1 / σ_i², so smaller-variance observations receive more weight.

## Equal variance case

If all variances are equal:

σ₁² = σ₂² = ... = σₙ² = σ²

then:

μ_hat
= [Σᵢ x_i / σ²] / [Σᵢ 1 / σ²]
= [(1 / σ²) Σᵢ x_i] / [n / σ²]
= Σᵢ x_i / n
= x_bar

So when all observations have the same variance, the weighted mean reduces to the ordinary sample mean.
