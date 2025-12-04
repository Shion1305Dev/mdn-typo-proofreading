resource "github_issue_label" "auto_generated" {
  repository  = github_repository.repo.name
  name        = "auto-generated"
  color       = "ededed"
  description = ""
}

resource "github_issue_label" "japanese_typo" {
  repository  = github_repository.repo.name
  name        = "japanese-typo"
  color       = "04291d"
  description = ""
}

resource "github_issue_label" "level_medium" {
  repository  = github_repository.repo.name
  name        = "level-medium"
  color       = "cfc7b4"
  description = ""
}

resource "github_issue_label" "level_low" {
  repository  = github_repository.repo.name
  name        = "level-low"
  color       = "4b5abb"
  description = ""
}

resource "github_issue_label" "non_japanese_typo" {
  repository  = github_repository.repo.name
  name        = "non-japanese-typo"
  color       = "9a1f67"
  description = ""
}

resource "github_issue_label" "level_high" {
  repository  = github_repository.repo.name
  name        = "level-high"
  color       = "cd5e08"
  description = ""
}
