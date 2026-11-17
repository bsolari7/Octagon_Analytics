"""Octagon Analytics: interactive final-project website for DATA 3301."""
from __future__ import annotations
import os
import pandas as pd
import streamlit as st
from modeling import QUESTIONS, load_artifacts, predict_matchup
from video_analysis import analyze_fight_video

# Replace these four entries before submitting.
TEAM_MEMBERS = [
    {"name":"Cis Garcia","email":"cgarc283@calpoly.edu","focus":"YOLO feasibility study and website","sections":"Footage analysis, Limitations"},
    {"name":"Brandon Solari","email":"bsolari@calpoly.edu","focus":"Data Visualization","sections":"EDA, Matchup predictor"},
    {"name":"Noah Nayve","email":"nnayve@calpoly.edu","focus":"Data preparation and provenance","sections":"Data, Decisions"},
    {"name":"David Morfin","email":"dmorfi04@calpoly.edu","focus":"Classification","sections":"Predictive questions, Conclusions"},
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
    st.caption("Source repositories: 1. https://github.com/jansen88/ufc-data/tree/master/data 2. https://github.com/komaksym/UFC-DataLab/tree/main/data")
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
    st.title("Decisions, Reflection, and Conclusions")

    st.header("Decisions and Reflection")
    st.write(
        "The summary below gives the major project choices at a glance. "
        "Open a decision card for the alternatives, reasoning, future revision, "
        "and AI-checking details."
    )

    decisions = [
        {
            "Decision": "Data sources",
            "Alternatives": (
                "Use one public UFC dataset; rely on betting odds; collect a "
                "new video dataset before modeling."
            ),
            "Choice": (
                "Combined detailed UFC bout statistics with fighter biography "
                "data. Excluded betting odds from the main models."
            ),
            "Why": (
                "Fight history plus age, height, reach, and weight adds pre-fight "
                "context while keeping the model focused on fighter information "
                "rather than market expectations."
            ),
            "Again?": (
                "Yes. We would add independently collected pre-fight information "
                "such as rankings, camp changes, or weigh-in data."
            ),
            "AI check": (
                "AI suggested combining fight-history and biography data. We checked "
                "normalized name joins and reviewed missing values after merging."
            ),
        },
        {
            "Decision": "Join key and preparation",
            "Alternatives": (
                "Join using raw names; manually match every name; drop all rows "
                "missing biography information."
            ),
            "Choice": (
                "Created normalized fighter-name keys by standardizing case, "
                "punctuation, accents, and repeated spaces."
            ),
            "Why": (
                "Raw names were inconsistent across files. Normalized keys retained "
                "more usable records and produced a more reliable merge."
            ),
            "Again?": (
                "Yes. With more time, we would manually audit ambiguous matches "
                "and use permanent fighter IDs if available."
            ),
            "AI check": (
                "AI suggested name normalization. We inspected unmatched and missing "
                "biography fields after the merge."
            ),
        },
        {
            "Decision": "Prediction targets",
            "Alternatives": (
                "Predict only winner; predict exact method; predict continuous "
                "fight duration."
            ),
            "Choice": (
                "Used five classification targets: winner, finish, KO/TKO, "
                "submission, and a 10-minute duration threshold."
            ),
            "Why": (
                "The targets share a pre-fight UFC-prediction theme, are "
                "interpretable, and can be consistently defined from the data."
            ),
            "Again?": (
                "Mostly yes. We would consider a duration-regression model after "
                "evaluating its error carefully."
            ),
            "AI check": (
                "AI suggested target definitions. We verified that each target "
                "could be created without using future information."
            ),
        },
        {
            "Decision": "Leakage prevention",
            "Alternatives": (
                "Use same-fight statistics; use career averages including the "
                "current fight; use lagged pre-fight averages."
            ),
            "Choice": (
                "Created record, recent-form, and performance features using only "
                "bouts before the bout being predicted."
            ),
            "Why": (
                "Same-fight statistics are only known after the outcome and would "
                "make the model unrealistically strong."
            ),
            "Again?": (
                "Yes. Leakage prevention is essential for a realistic prediction model."
            ),
            "AI check": (
                "AI suggested shifting statistics by one fight. We checked that "
                "historical averages are calculated after `shift()`."
            ),
        },
        {
            "Decision": "Train/test split",
            "Alternatives": (
                "Random split; cross-validation only; chronological split."
            ),
            "Choice": (
                "Used the earliest 80% of event dates for training and newest "
                "20% for held-out testing."
            ),
            "Why": (
                "This better matches the real task of using past fights to predict "
                "later fights."
            ),
            "Again?": (
                "Yes. With more time, we would add rolling time-based cross-validation."
            ),
            "AI check": (
                "AI suggested a time-based split. We confirmed that every test date "
                "occurs on or after the cutoff."
            ),
        },
        {
            "Decision": "Models and baseline",
            "Alternatives": (
                "Use one model; use a complex deep-learning model; compare a "
                "baseline, Logistic Regression, and Gradient Boosting."
            ),
            "Choice": (
                "Compared a constant-probability baseline with Logistic Regression "
                "and Gradient Boosting for every question."
            ),
            "Why": (
                "The baseline shows whether a model learns beyond the historical rate. "
                "Logistic Regression is interpretable, while Gradient Boosting can "
                "capture nonlinear patterns."
            ),
            "Again?": (
                "Yes. We would tune hyperparameters with time-aware validation "
                "before trying more advanced models."
            ),
            "AI check": (
                "AI suggested the model types and metrics. We retained them because "
                "the team can explain their purpose and compared them on the same test period."
            ),
        },
        {
            "Decision": "YOLO pose analysis",
            "Alternatives": (
                "Add pose metrics directly to winner model; exclude video; treat "
                "pose analysis as exploratory."
            ),
            "Choice": (
                "Kept YOLO pose analysis separate from the outcome-prediction models."
            ),
            "Why": (
                "We do not yet have a large, consistently labeled collection of "
                "fight videos with pre-fight pose features."
            ),
            "Again?": (
                "Yes. The next step is to build a labeled video dataset and test "
                "whether pose features improve held-out prediction."
            ),
            "AI check": (
                "AI suggested a pose-analysis tab. We label its measures as "
                "exploratory 2D estimates, not validated model features."
            ),
        },
    ]

    summary = pd.DataFrame(decisions)[
        ["Decision", "Choice", "Again?"]
    ].rename(
        columns={"Again?": "Would we make this choice again?"}
    )

    st.subheader("Table 2. Decision Summary")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.subheader("Decision Details")

    for item in decisions:
        with st.expander(item["Decision"]):
            st.markdown(f"**Choice**  \n{item['Choice']}")
            st.markdown(
                f"**Alternatives considered**  \n{item['Alternatives']}"
            )
            st.markdown(f"**Why we chose it**  \n{item['Why']}")
            st.markdown(
                f"**Would we make this choice again?**  \n{item['Again?']}"
            )
            st.markdown(
                f"**AI suggestion and how we checked it**  \n{item['AI check']}"
            )

    st.header("Discussion and Conclusions")

    st.subheader("What the Five Questions Say Together")
    st.write(
        "All five questions examine one common theme: how much publicly available "
        "pre-fight UFC information can tell us about a future bout. Historical "
        "record, recent form, physical characteristics, and prior performance "
        "contain useful signal, but the results are not strong enough to treat "
        "an individual probability as certain."
    )

    st.subheader("What We Would Tell a Potential User")
    st.write(
        "This website is best used as an exploratory matchup tool. It summarizes "
        "historical differences between fighters and estimates probabilities from "
        "past UFC patterns. It is not betting advice or a guarantee because important "
        "factors are missing, including injuries, training camp, weight cuts, game "
        "plans, late opponent changes, judging, and changing skill over time."
    )

    st.subheader("What We Learned About Predictive Data Science")
    st.write(
        "The key lesson was that a model can look useful while still being unrealistic "
        "if future information leaks into training. Lagged pre-fight features and a "
        "chronological holdout made the evaluation more credible. We also learned that "
        "a baseline is necessary and that accuracy alone can be misleading for rare "
        "outcomes such as submissions. New technology, including YOLO pose estimation, "
        "only helps if its features are consistently collected and evaluated on data "
        "the model has never seen."
    )

    st.subheader("Next Steps")
    st.markdown(
        """
        - Add verified pre-fight rankings, weigh-in information, and injury context.
        - Use rolling time-based cross-validation.
        - Audit fighter-name matches and seek stable fighter IDs.
        - Build a consistently labeled fight-video dataset before using pose features in prediction.
        - Add a held-out ground-truth page comparing pre-cutoff predictions with actual outcomes.
        """
    )

if page=="Overview":overview()
elif page=="Data and EDA":data_eda()
elif page=="Predictive questions":predictive_questions()
elif page=="Matchup predictor":matchup_predictor()
elif page=="Footage analysis":footage_analysis()
else:decisions_conclusion()
