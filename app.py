"""Octagon Analytics: interactive final-project website for DATA 3301."""
from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from modeling import QUESTIONS, load_artifacts, predict_matchup
from video_analysis import analyze_fight_video

# Replace these four entries before submitting.
TEAM_MEMBERS = [
    {"name":"Team member 1","email":"email@calpoly.edu","focus":"Data preparation and provenance","sections":"Data; Decisions and reflection"},
    {"name":"Team member 2","email":"email@calpoly.edu","focus":"Exploratory data analysis","sections":"Data and EDA"},
    {"name":"Team member 3","email":"email@calpoly.edu","focus":"Predictive modeling and evaluation","sections":"Predictive questions; Results"},
    {"name":"Team member 4","email":"email@calpoly.edu","focus":"YOLO feasibility study and website","sections":"Footage analysis; Limitations"},
]
PROJECT_TITLE = "Octagon Analytics: What Can Historical UFC Data Predict?"
st.set_page_config(page_title="Octagon Analytics", page_icon="🥊", layout="wide")
st.markdown("""<style>.block-container{max-width:1200px;padding-top:2rem}.hero{background:linear-gradient(110deg,#121826,#9b111e);border-radius:20px;color:#fff;padding:2.2rem 2.5rem;margin-bottom:1.4rem}.hero h1{margin:0;font-size:2.35rem}.hero p{opacity:.9;margin:.55rem 0 0}</style>""",unsafe_allow_html=True)
@st.cache_resource(show_spinner="Preparing pre-fight features and evaluating all five questions...")
def artifacts(): return load_artifacts()
data=artifacts()
st.sidebar.title("Octagon Analytics")
page=st.sidebar.radio("Report navigation",["Overview","Data and EDA","Predictive questions","Matchup predictor","Footage analysis","Decisions and conclusion"])
st.sidebar.caption("DATA 3301 final project - website report")
def pct(x): return "Not available" if pd.isna(x) else f"{x:.1%}"

def overview():
    st.markdown(f"<div class='hero'><h1>{PROJECT_TITLE}</h1><p>An interactive DATA 3301 report using historical UFC results, fighter biographies, and an exploratory pose-analysis workflow.</p></div>",unsafe_allow_html=True)
    st.header("Project team and focus areas")
    st.dataframe(pd.DataFrame(TEAM_MEMBERS).rename(columns={"name":"Team member","email":"Email","focus":"Focus area","sections":"Report section(s)"}),hide_index=True,use_container_width=True)
    st.warning("Before submitting, replace the four placeholder names, emails, focus areas, and section references in `app.py`.")
    st.header("Introduction")
    st.markdown("""UFC outcomes are difficult to predict because opponents differ in style, experience, size, recent form, and context that is not directly observed in public records. This project asks what can be predicted from information that would have been available **before** a bout. We combine historical UFC bout statistics with fighter biography fields to create lagged, pre-fight features and evaluate models chronologically.

Our primary question is which fighter is more likely to win a hypothetical matchup. We also test four bout-level questions about finish type and duration. Logistic regression provides an interpretable benchmark, while gradient boosting allows limited nonlinear relationships. The interactive matchup tool applies only the primary winner model; the remaining questions are evaluated as report results.

Across questions, the site emphasizes held-out performance and uncertainty rather than treating a probability as a guarantee. The footage tab is a separate feasibility study: YOLO can estimate 2D pose from uploaded clips, but those video metrics are not yet validated features in the outcome models.""")
    st.header("Questions and headline results")
    results=data["scores"].copy(); best=results.loc[results.groupby("Question")["ROC AUC"].idxmax(),["Question","Approach","ROC AUC","Accuracy"]].copy()
    best["ROC AUC"]=best["ROC AUC"].map(lambda x:f"{x:.3f}");best["Accuracy"]=best["Accuracy"].map(lambda x:f"{x:.3f}")
    st.dataframe(best.rename(columns={"Approach":"Best held-out approach"}),hide_index=True,use_container_width=True)
    st.info("Higher ROC AUC indicates better ranking of likely positive cases; accuracy alone can be misleading, particularly for rarer events such as submissions.")

def data_eda():
    st.title("Data and exploratory analysis"); d=data["diagnostics"]; a,b,c,e=st.columns(4)
    a.metric("Raw fight rows",f"{d['Raw fight rows']:,}");b.metric("Completed W/L fights",f"{d['Completed W/L fights retained']:,}");c.metric("Unique fighters",f"{d['Unique fighters']:,}");e.metric("Model cutoff",f"{d['Chronological cutoff']:%b %Y}")
    st.subheader("Provenance and structure")
    sources=pd.DataFrame([["merged_stats_n_scorecards.csv","One completed UFC bout, with red/blue results and in-fight statistics","Historical bout outcomes and detailed statistics","All five outcome targets and lagged career features"],["fighters_raw.csv","One fighter profile","Height, reach, listed weight, stance, DOB","Joined by normalized fighter name"],["events_raw.csv, odds_raw.csv, SCORECARDS.csv","Event, market, and scorecard records","Supplementary project sources","Retained for exploration; odds excluded from primary prediction"]],columns=["File","Observational unit","Contents","Role in report"])
    st.dataframe(sources,hide_index=True,use_container_width=True)
    st.caption("Source repository: https://github.com/Octagon-Analytics/Octagon_Analytics. Add exact raw-data URLs and citations before submission.")
    with st.expander("Data integration and preparation"):
        st.markdown("""1. Retained records with valid event dates and W/L outcomes; draws, no contests, and incomplete outcomes were excluded.  
2. Converted each bout into red- and blue-fighter perspectives. Strings such as `12 of 30` became landed and attempted values; control time became seconds.  
3. Sorted each fighter chronologically. Wins, recent form, and performance averages were shifted one bout backward so same-fight statistics cannot leak into prediction.  
4. Joined biography data using normalized names. Missing numeric features are imputed using training-set medians and an indicator flag.  
5. Used earliest 80% of event dates for training and newest 20% for testing. Both perspectives of a bout stay in the same period.""")
    st.subheader("Figure 1. Completed UFC bouts by year");st.bar_chart(data["valid"].assign(year=lambda x:x.event_date.dt.year).groupby("year").size())
    st.subheader("Figure 2. Recorded method distribution");methods=data["valid"].method.fillna("Missing").replace({"U-DEC":"Decision","S-DEC":"Decision","M-DEC":"Decision"}).value_counts().head(10);st.bar_chart(methods)
    st.subheader("Figure 3. Pre-fight win-rate advantage and observed win rate")
    fighter=data["fighter"].copy();fighter["bin"]=pd.cut(fighter.pre_win_pct_difference,bins=12);binned=fighter.groupby("bin",observed=True).agg(win_rate=("won_fight","mean")).reset_index();binned["advantage midpoint"]=binned["bin"].map(lambda x:x.mid);st.line_chart(binned.set_index("advantage midpoint")[["win_rate"]])
    st.caption("The pattern is descriptive, not causal: record advantage is related to many other differences between fighters.")
    with st.expander("Missingness and data-quality notes"):
        st.dataframe(data["missing"].head(18),hide_index=True,use_container_width=True);st.markdown("Key limitations include incomplete biography fields, inconsistent name spellings, missing in-fight values, public-record coverage, and unobserved camp, injury, weigh-in, or strategy context.")

def predictive_questions():
    st.title("Predictive questions: design, implementation, and results");st.write("All approaches use the same chronological holdout. Each question compares a constant-probability baseline, Logistic Regression, and Gradient Boosting on held-out data.")
    results=data["scores"].copy()
    for column in ["Accuracy","ROC AUC","Brier score"]:results[column]=results[column].map(lambda x:f"{x:.3f}")
    results["Positive rate"]=results["Positive rate"].map(lambda x:f"{x:.1%}")
    st.subheader("Table 1. Held-out model comparison");st.dataframe(results,hide_index=True,use_container_width=True);st.caption("ROC AUC measures probability ranking; Brier score measures probability accuracy, with lower values better. The baseline predicts the training positive rate for every test observation.")
    for key,detail in QUESTIONS.items():
        with st.expander(detail["title"],expanded=(key=="win")):
            st.markdown(f"**Question and target.** {detail['description']} The positive class is **{detail['target']}**.")
            st.markdown("**Features.** " + ("Fighter-minus-opponent differences in prior UFC bouts, smoothed win rate, recent form, age, size, and lagged striking/grappling/control averages." if key=="win" else "Pre-fight red/blue values plus their differences for the same record, form, size, and lagged performance variables. No same-fight statistic is used as a predictor."))
            st.markdown("**Design and implementation.** Time-based splitting simulates predicting later bouts from earlier data. Median imputation and scaling occur inside model pipelines after splitting to prevent test-data leakage.")
            st.dataframe(results.loc[results.Question.eq(detail["title"])],hide_index=True,use_container_width=True)
            st.markdown("**Interpretation and limits.** These results estimate prediction from available historical UFC records only. They cannot establish causes and may shift across eras or competition levels.")
    st.subheader("Approaches tried and retained");st.markdown("The baseline tests whether models beat a constant historical rate. Logistic Regression is the interpretable linear benchmark. Gradient Boosting captures modest nonlinear effects and interactions. Betting odds were considered but excluded from the primary models because the project aims to evaluate athlete-history information rather than reproduce market expectations.")

def matchup_predictor():
    st.title("Interactive matchup predictor");st.caption("All five questions use the selected fighters and only data from before the selected date.")
    names=data["names"].name.tolist();ia=names.index("Islam Makhachev") if "Islam Makhachev" in names else 0;ib=names.index("Charles Oliveira") if "Charles Oliveira" in names else 1
    with st.form("matchup"):
        left,right,date=st.columns([1,1,.75]);a=left.selectbox("Fighter A",names,index=ia);b=right.selectbox("Fighter B",names,index=ib);event_date=date.date_input("Hypothetical fight date",value=pd.Timestamp.today().date());st.form_submit_button("Predict matchup",type="primary",use_container_width=True)
    if a==b:st.warning("Choose two different fighters.");return
    prediction=predict_matchup(a,b,event_date);st.subheader(f"{a} vs. {b}")
    model_choice=st.radio("Prediction model",["Logistic regression","Gradient boosting"],horizontal=True,help="The report compares both approaches on a chronological held-out test set. Choose either approach to see all five live probabilities.")
    winner,finish,ko,submission,long=st.tabs(["1. Winner","2. Finish","3. KO/TKO","4. Submission","5. Duration"])
    with winner:
        p=prediction["winner"][model_choice];x,y,z=st.columns([1,.4,1]);x.metric(a,pct(p),"estimated win probability");y.markdown("### vs.");z.metric(b,pct(1-p),"estimated win probability");st.progress(int(p*100),text=f"{a}: {p:.1%}   |   {b}: {1-p:.1%}");st.caption("Positive outcome: Fighter A wins the bout.")
    for tab,key,positive,negative in [(finish,"finish","Bout ends before a decision","Bout goes to a decision"),(ko,"ko","Bout ends by KO/TKO","Bout does not end by KO/TKO"),(submission,"submission","Bout ends by submission","Bout does not end by submission"),(long,"long","Bout lasts at least 10 minutes","Bout lasts under 10 minutes")]:
        with tab:
            p=prediction["bout"][key][model_choice];left,right=st.columns(2);left.metric(positive,pct(p));right.metric(negative,pct(1-p));st.progress(int(p*100),text=f"{positive}: {p:.1%}   |   {negative}: {1-p:.1%}");st.caption(f"Question {QUESTIONS[key]['title'].split('.')[0]}. These are bout-level probabilities, not fighter-specific outcome probabilities.")
    labels={"pre_bouts":"UFC bouts","pre_win_pct":"Smoothed win rate","last_3_win_pct":"Last 3 fight win rate","age":"Age","height_inches":"Height (in.)","reach_inches":"Reach (in.)","weight_lbs":"Listed weight (lb.)","pre_sig_str_landed_avg":"Avg. significant strikes landed","pre_td_landed_avg":"Avg. takedowns landed","pre_sub_att_avg":"Avg. submission attempts","pre_ctrl_seconds_avg":"Avg. control time (sec.)"}
    profile=pd.DataFrame({"Feature":labels.values(),a:[prediction["a"].get(x) for x in labels],b:[prediction["b"].get(x) for x in labels]})
    for col in [a,b]:profile[col]=profile[col].map(lambda x:"Not available" if pd.isna(x) else f"{x:.2f}")
    st.subheader("Pre-fight profile used by the models");st.dataframe(profile,hide_index=True,use_container_width=True);st.warning("Educational estimate from historical patterns, not betting advice or a guaranteed outcome.")

def footage_analysis():
    st.title("YOLO pose feasibility study");st.markdown("This page tests whether usable 2D pose information can be extracted from fight video. It is not connected to outcome models because the project does not have a consistently labeled, held-out fight-video dataset.")
    clip=st.file_uploader("Upload a short fight clip",type=["mp4","mov","avi","mkv"]);c1,c2,c3=st.columns(3);sample=c1.select_slider("Pose samples per second",options=[1,2,3,5],value=2);duration=c2.select_slider("Maximum seconds to analyze",options=[15,30,60,120],value=60);confidence=c3.select_slider("Detection confidence",options=[.3,.4,.5,.6],value=.5)
    if clip and st.button("Analyze video with YOLO Pose",type="primary"):
        try:
            with st.spinner("Analyzing video..."):st.session_state["pose"]=analyze_fight_video(clip.getvalue(),os.path.splitext(clip.name)[1],sample,duration,confidence);st.session_state["pose_name"]=f"{os.path.splitext(clip.name)[0]}_pose_overlay.mp4"
        except Exception as error:st.error(f"Video analysis could not run: {error}")
    result=st.session_state.get("pose")
    if result:
        a,b,c=st.columns(3);a.metric("Seconds analyzed",f"{result['duration']:.1f}");b.metric("Pose samples",result["sampled"]);c.metric("Detection rate",f"{result['detection_rate']:.0%}");st.video(result["video"]);st.download_button("Download pose-overlay video",result["video"],st.session_state.get("pose_name","pose_overlay.mp4"),"video/mp4")
        report=result["subjects"].copy()
        for col in report.columns[2:]:report[col]=report[col].map(lambda x:"Not available" if pd.isna(x) else f"{x:.2f}")
        st.dataframe(report,hide_index=True,use_container_width=True)
    st.info("2D angles and movement depend on camera angle, zoom, occlusion, and temporary tracker-ID swaps. A next study would label many clips, extract the same features, and test whether they improve held-out performance beyond historical statistics.")

def decisions_conclusion():
    st.title("Decisions, reflection, and conclusions");st.subheader("Table 2. Major project decisions")
    decisions=pd.DataFrame([["Primary source","Use one public dataset only","Combined bout statistics with fighter biographies","Adds pre-fight experience, size, and performance context","Add independently collected or verified video features"],["Prediction time","Use same-fight statistics","Lag all performance features by one bout","Same-fight statistics would leak the result","Use time-stamped rankings and weigh-in data"],["Test split","Random train/test split","Chronological 80/20 split","Resembles forecasting later bouts from earlier bouts","Use rolling-origin validation"],["Models","One model or unexamined defaults","Baseline, Logistic Regression, and shallow Gradient Boosting","Balances interpretability with nonlinear modeling","Tune with nested time-aware validation"],["Odds","Include betting odds","Exclude from primary models","Focuses on athlete history, not market knowledge","Compare an odds-inclusive benchmark"],["YOLO pose","Treat pose output as prediction-ready","Present it as a feasibility study","No labeled historical clip dataset supports valid testing","Create a consistent annotated video dataset"]],columns=["Decision","Alternatives considered","Choice","Why","What we would do next"])
    st.dataframe(decisions,hide_index=True,use_container_width=True)
    st.subheader("Reflection on AI-assisted work");st.markdown("AI assistance accelerated code organization and interface development. The team must verify every transformation, model setting, metric, and claim before submission. Every member should be able to explain chronological splitting, lagged features, median imputation, ROC AUC versus Brier score, Logistic Regression, Gradient Boosting, and why YOLO output is not yet part of the outcome model.")
    st.subheader("Conclusion");st.markdown("Historical UFC records contain some useful signal for ranking hypothetical matchup outcomes, but probabilities remain uncertain. The outcome models cannot observe many decisive factors, including health, training camp, weight cut, strategy, judging, and changes in skill over time. The strongest next step is to collect consistent pre-fight and video data, then test whether those additions improve performance on future held-out bouts.")
    st.subheader("Submission checklist");st.markdown("""- [ ] Replace team placeholders and verify every member understands their focus area.
- [ ] Add exact raw-data URLs and source citations.
- [ ] Review every figure and table with surrounding explanatory text.
- [ ] Run the website from a clean environment using `requirements.txt`.
- [ ] Put the live website URL on the first line of the website README.
- [ ] Submit one outer ZIP containing the website/report and a separate code-and-data folder.
- [ ] Give the instructor GitHub access and retain dropped/exploratory approaches in the code submission.""")

if page=="Overview":overview()
elif page=="Data and EDA":data_eda()
elif page=="Predictive questions":predictive_questions()
elif page=="Matchup predictor":matchup_predictor()
elif page=="Footage analysis":footage_analysis()
else:decisions_conclusion()
