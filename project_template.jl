using Pkg
Pkg.activate(".")

using PkgTemplates

t = Template(;
    user="wilkieolin",
    dir=".",
    authors="Wilkie Olin-Ammentorp",
    julia=v"1.9",
    plugins=[
        License(; name="MIT"),
        Git(; manifest=true, ssh=true),
        GitHubActions(; x64=true),
        Codecov(),
        Documenter{GitHubActions}(),
        Develop(),
    ],
)

print(t)