import logging
import os
import sys

from analysed_resume.app import process
from analysed_resume.grammer import grammer
from analysed_resume.parser import parser
from analysed_resume.xyz_resume import xyz_resume


def root(resume,jd):
    primer=parser(resume)
    if primer:
        parsed_project=primer['projects']
        parsed_skills=primer['skills']
        parsed_experience=primer['experience']
        if parsed_project and parsed_skills and parsed_experience:
            logging.info('Resume has been parsed')
            p=process()
            analyzed_skills=p.skill_specific(parsed_skills, jd)
            analyzed_project=p.project_specific(parsed_project,jd)
            if 'Invalid_JD' in analyzed_skills and 'Invalid_JD' in analyzed_project:            
                return 'Invalid_JD'
            pande= "\n Project : "+"\n" + parsed_project + " \n  Experience : \n" + parsed_experience
            analysed_grammer=grammer(pande)
            
            xyz_analysed=xyz_resume(pande)
            pointers={
                'project':analyzed_project,
                'skills':analyzed_skills,
                'grammer':analysed_grammer,
                'xyz':xyz_analysed,
                'parsed_exp':parsed_experience
            }
            return pointers
        return None
    return None
