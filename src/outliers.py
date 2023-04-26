def remove_readmitted_outlier(value):
  if value not in ['NO']:
    return 'YES'
  else:
    return value


def remove_admission_source_id_outlier(value):
  if value not in ['Emergency Room', 'Physician Referral']:
    return 'Other'
  else:
    return value


def remove_discharge_outlier(value):
  if value not in ['Discharged to home', 'Discharged/transferred to SNF', "Discharged/transferred to home with home health service"]:
    return 'Other'
  else:
    return value


def remove_admission_type_outlier(value):
  if value not in ['Emergency', 'Elective', "Urgent"]:
    return 'Other'
  else:
    return value

def remove_gender_outlier(value):
  if value not in ['Female', 'Male']:
    return 'Female'
  else:
    return value


def remove_race_outlier(value):
  if value not in ['Caucasian', 'AfricanAmerican']:
    return 'Other'
  else:
    return value