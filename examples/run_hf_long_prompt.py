# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from transformers import AutoModelForCausalLM, AutoTokenizer

from minference import MInference

import time

#prompt = "Hello, my name is"
prompt = """###\nArticle: In general, the details of Aristotle's life are not well-established. The biographies written in ancient times are often speculative and historians only agree on a few salient points.[C] Aristotle was born in 384 BC[D] in Stagira, Chalcidice,[2] about 55 km (34 miles) east of modern-day Thessaloniki.[3][4] He was the son of Nicomachus, the personal physician of King Amyntas of Macedon,[5] and Phaestis, a woman with origins from Chalcis, Euboea.[6] Nicomachus was said to have belonged to the medical guild of Asclepiadae and was likely responsible for Aristotle's early interest in biology and medicine.[7] Ancient tradition held that Aristotle's family descended from the legendary physician Asclepius and his son Machaon.[8] Both of Aristotle's parents died when he was still at a young age and Proxenus of Atarneus became his guardian.[9] Although little information about Aristotle's childhood has survived, he probably spent some time in the Macedonian capital, making his first connections with the Macedonian monarchy.


School of Aristotle in Mieza, Macedonia, Greece.
At the age of seventeen or eighteen, Aristotle moved to Athens to continue his education at Plato's Academy. He became distinguished as a researcher and lecturer, earning for himself the nickname "mind of the school" by his tutor Plato.In Athens, he probably experienced the Eleusinian Mysteries as he wrote when describing the sights one viewed at the Mysteries, "to experience is to learn" (παθεĩν μαθεĩν).[13] Aristotle remained in Athens for nearly twenty years before leaving in 348/47 BC after Plato's death.[14] The traditional story about his departure records that he was disappointed with the academy's direction after control passed to Plato's nephew Speusippus, although it is possible that the anti-Macedonian sentiments in Athens could have also influenced his decision.Aristotle left with Xenocrates to Assos in Asia Minor, where he was invited by his former fellow student Hermias of Atarneus; he stayed there for a few years and left around the time of Hermias' death.[E] While at Assos, Aristotle and his colleague Theophrastus did extensive research in botany and marine biology, which they later continued at the near-by island of Lesbos.During this time, Aristotle married Pythias, Hermias's adoptive daughter and niece, and had a daughter whom they also named Pythias


"Aristotle tutoring Alexander" (1895) by Jean Leon Gerome Ferris.
In 343/42 BC, Aristotle was invited to Pella by Philip II of Macedon in order to become the tutor to his thirteen-year-old son Alexander;[19] a choice perhaps influenced by the relationship of Aristotle's family with the Macedonian dynasty.[20] Aristotle taught Alexander at the private school of Mieza, in the gardens of the Nymphs, the royal estate near Pella.[21] Alexander's education probably included a number of subjects, such as ethics and politics,[22] as well as standard literary texts, like Euripides and Homer. It is likely that during Aristotle's time in the Macedonian court, other prominent nobles, like Ptolemy and Cassander, would have occasionally attended his lectures.[24] Aristotle encouraged Alexander toward eastern conquest, and his own attitude towards Persia was strongly ethnocentric. In one famous example, he counsels Alexander to be "a leader to the Greeks and a despot to the barbarians".Alexander's education under the guardianship of Aristotle likely lasted for only a few years, as at around the age of sixteen he returned to Pella and was appointed regent of Macedon by his father Philip. During this time, Aristotle is said to have gifted Alexander an annotated copy of the Iliad, which reportedly became one of Alexander's most prized possessions. Scholars speculate that two of Aristotle's now lost works, On kingship and On behalf of the Colonies, were composed by the philosopher for the young prince.After Philip II's assassination in 336 BC, Aristotle returned to Athens for the second and final time a year later.

As a metic, Aristotle could not own property in Athens and thus rented a building known as the Lyceum (named after the sacred grove of Apollo Lykeios), in which he established his own school.[30] The building included a gymnasium and a colonnade (peripatos), from which the school acquired the name Peripatetic.[31] Aristotle conducted courses and research at the school for the next twelve years. He often lectured small groups of distinguished students and, along with some of them, such as Theophrastus, Eudemus, and Aristoxenus, Aristotle built a large library which included manuscripts, maps, and museum objects.[32] While in Athens, his wife Pythias died and Aristotle became involved with Herpyllis of Stagira. They had a son whom Aristotle named after his father, Nicomachus. This period in Athens, between 335 and 323 BC, is when Aristotle is believed to have composed many of his philosophical works. He wrote many dialogues, of which only fragments have survived. Those works that have survived are in treatise form and were not, for the most part, intended for widespread publication; they are generally thought to be lecture aids for his students. His most important treatises include Physics, Metaphysics, Nicomachean Ethics, Politics, On the Soul and Poetics. Aristotle studied and made significant contributions to "logic, metaphysics, mathematics, physics, biology, botany, ethics, politics, agriculture, medicine, dance, and theatre."


Portrait bust of Aristotle; an Imperial Roman (1st or 2nd century AD) copy of a lost bronze sculpture made by Lysippos.
While Alexander deeply admired Aristotle, near the end of his life, the two men became estranged having diverging opinions over issues, like the optimal administration of city-states, the treatment of conquered populations, such as the Persians, and philosophical questions, like the definition of braveness. A widespread speculation in antiquity suggested that Aristotle played a role in Alexander's death, but the only evidence of this is an unlikely claim made some six years after the death. Following Alexander's death, anti-Macedonian sentiment in Athens was rekindled. In 322 BC, Demophilus and Eurymedon the Hierophant reportedly denounced Aristotle for impiety,[38] prompting him to flee to his mother's family estate in Chalcis, Euboea, at which occasion he was said to have stated "I will not allow the Athenians to sin twice against philosophy"– a reference to Athens's trial and execution of Socrates. He died in Chalcis, Euboea of natural causes later that same year, having named his student Antipater as his chief executor and leaving a will in which he asked to be buried next to his wife. Aristotle left his works to Theophrastus, his successor as the head of the Lyceum, who in turn passed them down to Neleus of Scepsis in Asia Minor. There, the papers remained hidden for protection until they were purchased by the collector Apellicon. In the meantime, many copies of Aristotle's major works had already begun to circulate and be used in the Lyceum of Athens, Alexandria, and later in Rome.

Theoretical philosophy
Logic
Main article: Term logic
Further information: Non-Aristotelian logic
With the Prior Analytics, Aristotle is credited with the earliest study of formal logic,[44] and his conception of it was the dominant form of Western logic until 19th-century advances in mathematical logic.[45] Kant stated in the Critique of Pure Reason that with Aristotle, logic reached its completion.

Organon
Main article: Organon

Plato (left) and Aristotle in Raphael's 1509 fresco, The School of Athens. Aristotle holds his Nicomachean Ethics and gestures to the earth, representing his view in immanent realism, whilst Plato gestures to the heavens, indicating his Theory of Forms, and holds his Timaeus.[47][48]
Most of Aristotle's work is probably not in its original form, because it was most likely edited by students and later lecturers. The logical works of Aristotle were compiled into a set of six books called the Organon around 40 BC by Andronicus of Rhodes or others among his followers.The books are:

Categories
On Interpretation
Prior Analytics
Posterior Analytics
Topics
On Sophistical Refutations
The order of the books (or the teachings from which they are composed) is not certain, but this list was derived from analysis of Aristotle's writings. It goes from the basics, the analysis of simple terms in the Categories, the analysis of propositions and their elementary relations in On Interpretation, to the study of more complex forms, namely, syllogisms (in the Analytics)[50][51] and dialectics (in the Topics and Sophistical Refutations). The first three treatises form the core of the logical theory stricto sensu: the grammar of the language of logic and the correct rules of reasoning. The Rhetoric is not conventionally included, but it states that it relies on the Topics.

One of Aristotle's types of syllogism[F]
In words	In
terms[G]	In equations[H]
    All men are mortal.

    All Greeks are men.

∴ All Greeks are mortal.	M a P

S a M

S a P	
What is today called Aristotelian logic with its types of syllogism (methods of logical argument),[53] Aristotle himself would have labelled "analytics". The term "logic" he reserved to mean dialectics.

Metaphysics
Main article: Metaphysics (Aristotle)
The word "metaphysics" appears to have been coined by the first century AD editor who assembled various small selections of Aristotle's works to create the treatise we know by the name Metaphysics.[55] Aristotle called it "first philosophy", and distinguished it from mathematics and natural science (physics) as the contemplative (theoretikē) philosophy which is "theological" and studies the divine. He wrote in his Metaphysics (1026a16):

If there were no other independent things besides the composite natural ones, the study of nature would be the primary kind of knowledge; but if there is some motionless independent thing, the knowledge of this precedes it and is first philosophy, and it is universal in just this way, because it is first. And it belongs to this sort of philosophy to study being as being, both what it is and what belongs to it just by virtue of being.

Substance
Further information: Hylomorphism
Aristotle examines the concepts of substance (ousia) and essence (to ti ên einai, "the what it was to be") in his Metaphysics (Book VII), and he concludes that a particular substance is a combination of both matter and form, a philosophical theory called hylomorphism. In Book VIII, he distinguishes the matter of the substance as the substratum, or the stuff of which it is composed. For example, the matter of a house is the bricks, stones, timbers, etc., or whatever constitutes the potential house, while the form of the substance is the actual house, namely 'covering for bodies and chattels' or any other differentia that let us define something as a house. The formula that gives the components is the account of the matter, and the formula that gives the differentia is the account of the form.

Immanent realism
Main article: Aristotle's theory of universals

Plato's forms exist as universals, like the ideal form of an apple. For Aristotle, both matter and form belong to the individual thing (hylomorphism).
Like his teacher Plato, Aristotle's philosophy aims at the universal. Aristotle's ontology places the universal (katholou) in particulars (kath' hekaston), things in the world, whereas for Plato the universal is a separately existing form which actual things imitate. For Aristotle, "form" is still what phenomena are based on, but is "instantiated" in a particular substance.[55]

Plato argued that all things have a universal form, which could be either a property or a relation to other things. When one looks at an apple, for example, one sees an apple, and one can also analyse a form of an apple. In this distinction, there is a particular apple and a universal form of an apple. Moreover, one can place an apple next to a book, so that one can speak of both the book and apple as being next to each other. Plato argued that there are some universal forms that are not a part of particular things. For example, it is possible that there is no particular good in existence, but "good" is still a proper universal form. Aristotle disagreed with Plato on this point, arguing that all universals are instantiated at some period of time, and that there are no universals that are unattached to existing things. In addition, Aristotle disagreed with Plato about the location of universals. Where Plato spoke of the forms as existing separately from the things that participate in them, Aristotle maintained that universals exist within each thing on which each universal is predicated. So, according to Aristotle, the form of apple exists within each apple, rather than in the world of the forms.[55][58]

Potentiality and actuality
Concerning the nature of change (kinesis) and its causes, as he outlines in his Physics and On Generation and Corruption (319b–320a), he distinguishes coming-to-be (genesis, also translated as 'generation') from:

growth and diminution, which is change in quantity;
locomotion, which is change in space; and
alteration, which is change in quality.

Aristotle argued that a capability like playing the flute could be acquired – the potential made actual – by learning.
Coming-to-be is a change where the substrate of the thing that has undergone the change has itself changed. In that particular change he introduces the concept of potentiality (dynamis) and actuality (entelecheia) in association with the matter and the form. Referring to potentiality, this is what a thing is capable of doing or being acted upon if the conditions are right and it is not prevented by something else. For example, the seed of a plant in the soil is potentially (dynamei) a plant, and if it is not prevented by something, it will become a plant. Potentially, beings can either 'act' (poiein) or 'be acted upon' (paschein), which can be either innate or learned. For example, the eyes possess the potentiality of sight (innate – being acted upon), while the capability of playing the flute can be possessed by learning (exercise – acting). Actuality is the fulfilment of the end of the potentiality. Because the end (telos) is the principle of every change, and potentiality exists for the sake of the end, actuality, accordingly, is the end. Referring then to the previous example, it can be said that an actuality is when a plant does one of the activities that plants do.[55]

For that for the sake of which (to hou heneka) a thing is, is its principle, and the becoming is for the sake of the end; and the actuality is the end, and it is for the sake of this that the potentiality is acquired. For animals do not see in order that they may have sight, but they have sight that they may see.[59]

In summary, the matter used to make a house has potentiality to be a house and both the activity of building and the form of the final house are actualities, which is also a final cause or end. Then Aristotle proceeds and concludes that the actuality is prior to potentiality in formula, in time and in substantiality. With this definition of the particular substance (i.e., matter and form), Aristotle tries to solve the problem of the unity of the beings, for example, "what is it that makes a man one"? Since, according to Plato there are two Ideas: animal and biped, how then is man a unity? However, according to Aristotle, the potential being (matter) and the actual one (form) are one and the same.
Epistemology
Aristotle's immanent realism means his epistemology is based on the study of things that exist or happen in the world, and rises to knowledge of the universal, whereas for Plato epistemology begins with knowledge of universal Forms (or ideas) and descends to knowledge of particular imitations of these.[52] Aristotle uses induction from examples alongside deduction, whereas Plato relies on deduction from a priori principles.[52]
\n\nSummarize the above article in several sentences.\n"""

model_name = "microsoft/Phi-3-mini-128k-instruct" #"gradientai/Llama-3-8B-Instruct-262k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda",
)

# Patch MInference Module
minference_patch = MInference("tri_shape", model_name, kv_type="snapkv", starting_layer=1, attn_kwargs={"n_local": 1024, "n_init": 128, "n_last": 128, "max_capacity_prompt": 1056})
model = minference_patch(model)

batch_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
start = time.time()
outputs = model.generate(**batch_inputs, max_new_tokens=128)
print("Elapsed: ", time.time() - start)
generated_text = tokenizer.decode(
    outputs[0][batch_inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
print(f"Generated text: {generated_text!r}")
