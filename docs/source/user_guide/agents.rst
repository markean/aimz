Coding Agents
=============

.. meta::
   :description: The skill for coding agents that ships inside the aimz package, and how to point an agent at it.

aimz ships a skill for coding agents inside the package, so the copy next to an installed aimz describes that version.
The skill is a `SKILL.md <https://github.com/markean/aimz/blob/main/aimz/.agents/skills/aimz/SKILL.md>`__ file in the ``.agents/skills/aimz`` directory of the package.
It tells an agent what aimz requires of a kernel, which defaults to use, how to express a scenario depending on where the treatment enters the kernel, how to keep two scenarios paired, which mistakes return a number without an error, and when not to report an effect.

For an agent without aimz installed, this documentation is also published as plain text in `llms.txt <../llms.txt>`__ and `llms-full.txt <../llms-full.txt>`__.


Pointing an Agent at the Skill
------------------------------

An agent does not find the skill on its own.
There are two ways to point it at the skill, and they can be combined.

For an agent that loads skills from a directory of the project, run `Library Skills <https://library-skills.io>`__ in a project that lists aimz as a dependency:

.. code-block:: sh

    uvx library-skills

It finds the skills that the installed dependencies ship and links the ones you select into the project, so the link follows the installed version of aimz.

For any other agent, or to have the skill named in every session, add a pointer to the instructions file that the agent reads in your project, such as ``AGENTS.md``:

.. code-block:: markdown

    ## aimz

    Before writing or reviewing code that uses aimz, read the skill shipped with the installed package:
    `python -c "from importlib.resources import files; print(files('aimz') / '.agents/skills/aimz/SKILL.md')"`
    Do not rely on memory of the aimz API; it changes between minor releases.

The command prints the path of the file in the environment it runs in.
