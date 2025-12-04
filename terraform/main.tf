terraform {
  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 6.0"
    }
  }
}

provider "github" {
  owner = "Shion1305"
}

resource "github_repository" "repo" {
  name           = "mdn-typo-proofreading"
  description    = "Project to help MDN team with proofreading"
  visibility     = "public"
  has_downloads  = true
  has_issues     = true
  has_projects   = true
  has_wiki       = true
}
