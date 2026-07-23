# Resume Job Search Automation

This automation searches for non-coding analyst roles that fit Manoj's resume, filters out developer-heavy titles, and saves Excel outputs with timestamps to:

- `C:\Users\MANO\Documents\job_search_alerts\job_matches_latest.xlsx`
- `C:\Users\MANO\Documents\job_search_alerts\history\job_matches_YYYYMMDD_HHMMSS.xlsx`

## Current targeting

- Location: `Bengaluru, Karnataka, India`
- Roles: `data analyst`, `scm analyst`
- Strong matches expanded into: supply chain, operations, reporting, MIS, procurement, inventory, and master data analyst roles
- Sites: `Indeed`, `Google Jobs`

## Files

- `resume_job_search.py`: main search, scoring, dedupe, and Excel export script
- `config.json`: editable search settings
- `install_jobspy.ps1`: installs `python-jobspy` into a local `.deps` folder
- `register_hourly_task.ps1`: creates an hourly Windows scheduled task

## Notes

- The script uses your newer resume by default: `C:\Users\MANO\Documents\Manoj Kumar T_RESUME.pdf`
- SQL and NoSQL are treated as acceptable skills, but coding-heavy roles are filtered out
- `hours_old` is set to `48` so hourly runs still catch jobs posted while the machine was offline
- `state_seen_jobs.csv` keeps track of already-seen job links so each run can highlight only fresh matches

## Manual run

```powershell
& "C:\Users\MANO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  "D:\mano_github_pvt\scripts\job_search_automation\resume_job_search.py"
```
